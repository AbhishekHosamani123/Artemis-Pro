package com.artemis.helper;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.AnimatorSet;
import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.RadialGradient;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.view.animation.AccelerateInterpolator;
import android.view.animation.OvershootInterpolator;

/**
 * On-screen "AI cursor" drawn by the accessibility service while Artemis drives
 * the device: a glowing blue dot with an "AI" tag that pops onto the screen at
 * every point Artemis is about to touch, with an expanding tap ring, so a human
 * watching the phone can follow exactly what the agent does — including when it
 * is driven over MCP from an IDE.
 *
 * The overlay uses {@link android.view.WindowManager.LayoutParams#TYPE_ACCESSIBILITY_OVERLAY},
 * which accessibility services may draw without any extra permission, and is
 * flagged not-touchable so it can never intercept the very tap it advertises.
 * One view instance is reused: a new {@link #show} moves it to the next target
 * and replays the entry animation, which reads as a cursor jumping between
 * targets rather than a stream of icons.
 */
public final class CursorOverlay {

    private static final String TAG = "ArtemisCursor";
    private static final long DEFAULT_DURATION_MS = 1100L;

    private static final Object LOCK = new Object();
    private static Handler mainHandler;
    private static CursorView activeView;

    private CursorOverlay() {}

    /** Show the AI cursor at device pixel coordinates (fire and forget). */
    public static void show(ArtemisAccessibilityService service, float x, float y, long durationMs) {
        if (service == null) return;
        final long duration = durationMs > 0 ? durationMs : DEFAULT_DURATION_MS;
        final Handler handler = handler();
        handler.post(new Runnable() {
            @Override
            public void run() {
                try {
                    showOnMainThread(service, x, y, duration);
                } catch (Throwable t) {
                    Log.w(TAG, "Cursor overlay failed: " + t.getMessage());
                }
            }
        });
    }

    /** Remove the overlay immediately (safe to call from any thread). */
    public static void hide() {
        handler().post(new Runnable() {
            @Override
            public void run() {
                try {
                    removeActive();
                } catch (Throwable t) {
                    Log.w(TAG, "Hiding the cursor overlay failed: " + t.getMessage());
                }
            }
        });
    }

    private static Handler handler() {
        synchronized (LOCK) {
            if (mainHandler == null) {
                mainHandler = new Handler(Looper.getMainLooper());
            }
            return mainHandler;
        }
    }

    private static void showOnMainThread(
            ArtemisAccessibilityService service, float x, float y, long durationMs) {
        final WindowManager wm =
                (WindowManager) service.getSystemService(Context.WINDOW_SERVICE);
        if (wm == null) return;

        final CursorView view;
        if (activeView != null && activeView.isAttachedToWindow()) {
            view = activeView;
            view.cancelAnimations();
            view.moveTo(x, y);
            try {
                wm.updateViewLayout(view, view.layoutParams());
            } catch (Exception e) {
                // The window may have been removed by the system; fall through
                // to a fresh add below.
                if (view.getParent() != null) {
                    wm.removeView(view);
                }
                activeView = null;
                view.forceLayout();
                showOnMainThread(service, x, y, durationMs);
                return;
            }
        } else {
            removeActive();
            view = new CursorView(service);
            view.moveTo(x, y);
            activeView = view;
            try {
                wm.addView(view, view.layoutParams());
            } catch (Exception e) {
                Log.w(TAG, "Could not add the AI cursor overlay: " + e.getMessage());
                activeView = null;
                return;
            }
        }
        view.playEntry(durationMs);
    }

    private static void removeActive() {
        if (activeView == null) return;
        final CursorView view = activeView;
        activeView = null;
        view.cancelAnimations();
        if (view.isAttachedToWindow()) {
            try {
                final WindowManager wm =
                        (WindowManager) view.getContext().getSystemService(Context.WINDOW_SERVICE);
                if (wm != null) wm.removeView(view);
            } catch (Exception ignored) {}
        }
    }

    // ------------------------------------------------------------------ //

    private static final class CursorView extends View {

        private static final int BLUE = 0xFF1E6FFF;
        private static final int BLUE_LIGHT = 0xFF6FB4FF;
        private static final int BLUE_DEEP = 0xFF0B41B8;

        private final float density;
        private final WindowManager.LayoutParams params;

        private final Paint glowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint discPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint ringPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint dotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint pillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint pillStroke = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint tagPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint ripplePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

        private final RectF pillRect = new RectF();

        private float cursorX;          // device pixel coordinates of the pointer
        private float cursorY;
        private float rippleProgress = -1f; // <0 hides the tap ring

        private AnimatorSet entryAnim;
        private ValueAnimator rippleAnim;
        private Animator exitAnim;
        private Runnable exitRunnable;

        CursorView(ArtemisAccessibilityService service) {
            super(service);
            density = service.getResources().getDisplayMetrics().density;
            final int size = px(132);
            cursorX = size / 2f;
            cursorY = size / 2f;

            params = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                            | WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                            | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
                            | WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
                    PixelFormat.TRANSLUCENT);
            params.gravity = Gravity.TOP | Gravity.START;
            params.width = size;
            params.height = size;
            params.x = 0;
            params.y = 0;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                // API 30+ windows fit the system bars by default, which would
                // push every y coordinate below the status bar. The cursor is
                // positioned in raw screen pixels; fit nothing.
                params.setFitInsetsTypes(0);
            }

            setupPaints();
        }

        WindowManager.LayoutParams layoutParams() {
            return params;
        }

        private int px(float dp) {
            return Math.round(dp * density);
        }

        private void setupPaints() {
            glowPaint.setStyle(Paint.Style.FILL);
            discPaint.setStyle(Paint.Style.FILL);
            ringPaint.setStyle(Paint.Style.STROKE);
            ringPaint.setStrokeWidth(px(3f));
            ringPaint.setColor(Color.WHITE);
            dotPaint.setStyle(Paint.Style.FILL);
            dotPaint.setColor(Color.WHITE);
            pillPaint.setStyle(Paint.Style.FILL);
            pillPaint.setColor(BLUE);
            pillStroke.setStyle(Paint.Style.STROKE);
            pillStroke.setStrokeWidth(px(1.4f));
            pillStroke.setColor(0xAAFFFFFF);
            tagPaint.setStyle(Paint.Style.FILL);
            tagPaint.setColor(Color.WHITE);
            tagPaint.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
            tagPaint.setTextAlign(Paint.Align.CENTER);
            tagPaint.setTextSize(px(11f));
            ripplePaint.setStyle(Paint.Style.STROKE);
            ripplePaint.setStrokeWidth(px(2.6f));
            ripplePaint.setColor(BLUE_LIGHT);
        }

        void moveTo(float deviceX, float deviceY) {
            cursorX = params.width / 2f;
            cursorY = params.height / 2f;
            params.x = Math.round(deviceX - params.width / 2f);
            params.y = Math.round(deviceY - params.height / 2f);
            invalidate();
        }

        void cancelAnimations() {
            if (entryAnim != null) { entryAnim.cancel(); entryAnim = null; }
            if (rippleAnim != null) { rippleAnim.cancel(); rippleAnim = null; }
            if (exitAnim != null) { exitAnim.cancel(); exitAnim = null; }
            if (exitRunnable != null) {
                removeCallbacks(exitRunnable);
                exitRunnable = null;
            }
        }

        /** Pop in (or re-pop at the new position), burst once, fade out after a beat. */
        void playEntry(final long durationMs) {
            cancelAnimations();
            rippleProgress = -1f;
            setAlpha(1f);
            setScaleX(0.2f);
            setScaleY(0.2f);
            invalidate();

            entryAnim = new AnimatorSet();
            final ValueAnimator pop = ValueAnimator.ofFloat(0.2f, 1f);
            pop.setDuration(240L);
            pop.setInterpolator(new OvershootInterpolator(3.2f));
            pop.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
                @Override
                public void onAnimationUpdate(ValueAnimator animation) {
                    float scale = (Float) animation.getAnimatedValue();
                    setScaleX(scale);
                    setScaleY(scale);
                }
            });
            entryAnim.play(pop);
            entryAnim.addListener(new AnimatorListenerAdapter() {
                @Override
                public void onAnimationEnd(Animator animation) {
                    entryAnim = null;
                }
            });
            entryAnim.start();
            playRipple();

            exitRunnable = new Runnable() {
                @Override
                public void run() {
                    exitRunnable = null;
                    playExit();
                }
            };
            postDelayed(exitRunnable, Math.max(300L, durationMs));
        }

        private void playRipple() {
            rippleAnim = ValueAnimator.ofFloat(0f, 1f);
            rippleAnim.setDuration(520L);
            rippleAnim.setInterpolator(new AccelerateInterpolator(1.4f));
            rippleAnim.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
                @Override
                public void onAnimationUpdate(ValueAnimator animation) {
                    rippleProgress = (Float) animation.getAnimatedValue();
                    invalidate();
                }
            });
            rippleAnim.addListener(new AnimatorListenerAdapter() {
                @Override
                public void onAnimationEnd(Animator animation) {
                    rippleProgress = -1f;
                    invalidate();
                    if (rippleAnim == animation) rippleAnim = null;
                }
            });
            rippleAnim.start();
        }

        private void playExit() {
            final ValueAnimator fade = ValueAnimator.ofFloat(1f, 0f);
            fade.setDuration(220L);
            fade.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
                @Override
                public void onAnimationUpdate(ValueAnimator animation) {
                    float alpha = (Float) animation.getAnimatedValue();
                    setAlpha(alpha);
                    float scale = 0.55f + 0.45f * alpha;
                    setScaleX(scale);
                    setScaleY(scale);
                }
            });
            fade.addListener(new AnimatorListenerAdapter() {
                private boolean cancelled;

                @Override
                public void onAnimationCancel(Animator animation) {
                    cancelled = true;
                }

                @Override
                public void onAnimationEnd(Animator animation) {
                    exitAnim = null;
                    if (cancelled || activeView != CursorView.this) return;
                    removeActive();
                }
            });
            exitAnim = fade;
            fade.start();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            final float cx = cursorX;
            final float cy = cursorY;
            final float core = px(15f);
            final float glow = px(30f);

            glowPaint.setShader(new RadialGradient(
                    cx, cy, glow * 1.8f,
                    new int[]{0x591E6FFF, 0x261E6FFF, 0x001E6FFF},
                    new float[]{0f, 0.55f, 1f},
                    Shader.TileMode.CLAMP));
            canvas.drawCircle(cx, cy, glow * 1.8f, glowPaint);

            if (rippleProgress >= 0f) {
                final float radius = core + (glow * 1.5f - core) * rippleProgress;
                final int alpha = Math.round(200f * (1f - rippleProgress));
                ripplePaint.setAlpha(alpha);
                canvas.drawCircle(cx, cy, radius, ripplePaint);
            }

            discPaint.setShader(new LinearGradient(
                    cx, cy - core, cx, cy + core,
                    BLUE_LIGHT, BLUE_DEEP,
                    Shader.TileMode.CLAMP));
            canvas.drawCircle(cx, cy, core, discPaint);
            canvas.drawCircle(cx, cy, core, ringPaint);
            canvas.drawCircle(cx, cy, px(4.5f), dotPaint);

            drawAiTag(canvas, cx, cy, core);
        }

        private void drawAiTag(Canvas canvas, float cx, float cy, float core) {
            final String text = "AI";
            final float textWidth = tagPaint.measureText(text);
            final float padH = px(7f);
            final float pillW = textWidth + padH * 2f;
            final float pillH = px(19f);
            final float left = cx + core + px(6f);
            final float top = cy - core - pillH + px(2f);
            pillRect.set(left, top, left + pillW, top + pillH);

            canvas.drawRoundRect(pillRect, pillH / 2f, pillH / 2f, pillPaint);
            canvas.drawRoundRect(pillRect, pillH / 2f, pillH / 2f, pillStroke);

            final float textY = pillRect.centerY() - (tagPaint.descent() + tagPaint.ascent()) / 2f;
            canvas.drawText(text, pillRect.centerX(), textY, tagPaint);
        }
    }
}
