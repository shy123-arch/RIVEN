"""Motion-code training spans; no dataset access or accelerator dependencies."""


def choose_span(length, window, mode, rng):
    """Return (start, frames), aligned to the tokenizer's four-frame stride.

    All modes consume the original random-window offset draw. With equal seeds,
    the NumPy sample order therefore stays identical across training controls.
    Full mode ignores the sampled offset/window and returns the complete motion.
    """
    if mode not in ('random', 'prefix', 'full'):
        raise ValueError('unknown motion span mode')
    n = int(length) // 4 * 4
    if n < 4 or int(window) < 4:
        raise ValueError('motion length and reference window must be at least four')
    w = max(4, min(int(window), n) // 4 * 4)
    sampled_start = int(rng.integers(0, n - w + 1)) // 4 * 4
    if mode == 'full':
        return 0, n
    return (sampled_start if mode == 'random' else 0), w
