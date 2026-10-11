"""Example 1 - the selective SSM block.

Demonstrates

* the parallel scan (training mode) and the O(1) ``step`` (streaming mode) are
  mathematically identical,
* the state layout and the Mamba-3 exp-trapezoidal recurrence,
* gradient flow through every parameter group,
* the multi-timescale stack.

Run: ``python examples/01_ssm_basics.py``
"""

import jax
import jax.numpy as jnp
from common import banner, section
from flax import nnx

from ssm_latent import MultiScaleSsmBlock, MultiScaleSsmConfig, SsmBlock, SsmConfig
from ssm_latent.utils import count_parameters, pretty_params


def main() -> None:
    banner("Mamba-3 selective SSM in JAX/Flax")

    cfg = SsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, use_conv=True, conv_kernel=4
    )
    block = SsmBlock(cfg, nnx.Rngs(0))
    section("configuration")
    print(cfg)
    print(f"d_inner={cfg.d_inner}  d_head={cfg.d_head}  d_head_mimo={cfg.d_head_mimo}")
    print(f"parameters: {count_parameters(block)}")

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, cfg.d_model))

    section("parallel scan vs O(1) streaming step")
    y_parallel = block.forward(x, scan_mode="parallel")
    y_sequential = block.forward(x, scan_mode="sequential")
    state = block.zero_state(batch=2)
    ys = []
    for t in range(x.shape[1]):
        y_t, state = block.step(x[:, t], state)
        ys.append(y_t)
    y_step = jnp.stack(ys, axis=1)

    print(
        f"max |parallel - sequential_scan| = {float(jnp.max(jnp.abs(y_parallel - y_sequential))):.3e}"
    )
    print(f"max |parallel - step_loop      | = {float(jnp.max(jnp.abs(y_parallel - y_step))):.3e}")
    print(f"streaming state: h{tuple(state.h.shape)}, conv{tuple(state.conv_state.shape)}")

    section("gradients")
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(block)

    paths = jax.tree_util.tree_flatten_with_path(nnx.state(grads, nnx.Param))[0]
    for path, value in paths[:6]:
        name = "/".join(
            str(getattr(k, "key", k)) for k in path if getattr(k, "key", None) != "value"
        )
        print(f"  d/d{name:<18} shape={tuple(jnp.shape(value))}")
    print(f"  ... ({len(pretty_params(block))} parameter tensors in total)")
    print("  all finite:", all(bool(jnp.all(jnp.isfinite(v))) for _, v in paths))

    section("multi-timescale stack (fast / medium / slow)")
    ms_cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    ms = MultiScaleSsmBlock(ms_cfg, nnx.Rngs(1))
    y_ms = ms(x)
    print(f"output {tuple(y_ms.shape)}, parameters {count_parameters(ms)}")
    for i, layer in enumerate(ms.layers):
        lo, hi = ms_cfg.timescale_range(i)
        print(
            f"  layer {i}: a_re mean={float(jnp.mean(layer.a_re[...])):+.4f} "
            f"(init range [{lo}, {hi}])  conv={'yes' if layer.conv_weight is not None else 'no'}"
        )
    print(f"max |parallel - step| = {float(jnp.max(jnp.abs(y_ms - ms.scan_sequential(x)))):.3e}")


if __name__ == "__main__":
    main()
