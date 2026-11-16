# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public contracts for runtime validation, patch provenance and CPU CI."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _python_runtime_asserts(tree):
    # CuTe decorators compile the function's source AST. Their assertions are
    # compiler constraints, not CPython runtime validation removed by python -O.
    # Constructors and ordinary helpers still require explicit exceptions.
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        parent = node
        while parent in parents:
            parent = parents[parent]
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                ast.unparse(decorator) in {"cute.jit", "cute.kernel"}
                for decorator in parent.decorator_list
            ):
                break
        else:
            result.append(node)
    return result


def test_runtime_validation_does_not_use_optimizable_asserts():
    violations = []
    for path in sorted((ROOT / "aspec").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(
            f"{path.relative_to(ROOT)}:{node.lineno}"
            for node in _python_runtime_asserts(tree)
        )
    assert not violations, (
        "Production validation must use explicit exceptions because python -O removes assert: "
        + ", ".join(violations)
    )


def test_cute_compile_asserts_do_not_exempt_host_validation():
    tree = ast.parse(
        "assert host_guard\n"
        "@cute.jit\ndef compiled():\n    assert compile_guard\n"
        "@cute.kernel\ndef kernel():\n    assert tile_guard\n"
        "def host():\n    assert runtime_guard\n"
    )
    assert [ast.unparse(node.test) for node in _python_runtime_asserts(tree)] == [
        "host_guard", "runtime_guard"
    ]


def test_runtime_remote_code_policy_never_defaults_to_trusted():
    violations = []
    for path in sorted((ROOT / "aspec").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if (
                        keyword.arg == "trust_remote_code"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 3
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "trust_remote_code"
                    and isinstance(node.args[2], ast.Constant)
                    and node.args[2].value is True
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = [*node.args.posonlyargs, *node.args.args]
                positional_defaults = [None] * (
                    len(positional) - len(node.args.defaults)
                ) + list(node.args.defaults)
                defaults = dict(
                    zip(
                        [arg.arg for arg in positional],
                        positional_defaults,
                        strict=True,
                    )
                )
                defaults.update(
                    zip(
                        [arg.arg for arg in node.args.kwonlyargs],
                        node.args.kw_defaults,
                        strict=True,
                    )
                )
                default = defaults.get("trust_remote_code")
                if isinstance(default, ast.Constant) and default.value is True:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, (
        "Production trust_remote_code policy must default to false: "
        + ", ".join(sorted(set(violations)))
    )


def test_current_sglang_patch_has_registered_provenance():
    references = (ROOT / "docs/framework/REFERENCES.md").read_text(encoding="utf-8")
    patch = ROOT / "aspec/compat/patches/sglang-v0.5.21.patch"
    assert patch.is_file()
    assert "e00930c5489053f26d86b179cee0d087f846acbb" in patch.read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "`aspec/compat/patches/sglang-v0.5.21.patch`" in references
    assert "SGLang v0.5.21" in references
    assert (ROOT / "licenses/SGLang-LICENSE").is_file()


def test_public_ci_runs_pytest():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python -m pytest" in workflow


def test_eagle4_validation_gates_do_not_use_optimizable_asserts():
    for name in ("eagle4_upstream.py", "eagle4_asset.py", "eagle4_candidate_state.py"):
        path = ROOT / "tools" / "validation" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name


def test_public_ci_includes_eagle4_cpu_contracts():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python -m pytest -q tests/cpu" in workflow
    for name in ("test_eagle4.py", "test_eagle4_moe.py", "test_eagle4_validation.py"):
        assert (ROOT / "tests/cpu" / name).is_file()
