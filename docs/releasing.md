# Releasing

ASpec publishes the Draft package `ling-aspec`, Target package `ling-aspec-target`, and
shared code package `ling-aspec-runtime` at the same version. Python imports and the CLI use `aspec`.

## Versioning

Maintainers set `project.version` in [pyproject.toml](../pyproject.toml).
Current prereleases use `<major>.<minor>.<patch>b0`; maintainers choose the base version.
Tags must match `v<version>` exactly. CI validates the version without changing it.
See the [Python version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/).

## Release checks

Run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md) and the relevant
[validation gates](contributing/validation.md) for the changes being released.
Build from a clean, reviewed commit:

```bash
python -m tools.build_release --outdir dist
```

The output directory must be empty. The command builds and audits the sdist and wheel,
includes the runtime requirement files, and records their checksums in `SHA256SUMS`.
Verify source attribution and license texts, and exclude credentials, datasets, weights,
checkpoints and local environment files.

The [release workflow](../.github/workflows/build-release.yml) checks package metadata,
builds the distributions, and validates the installed wheel outside the source checkout.
Manual runs perform these checks without publishing. CPU installation checks do not replace
the GPU validation required for runtime changes.

Wheels include the CLI, model configurations, recipes, Quickstart data, runtime requirement
files and the SGLang patch. Source distributions also include documentation, tests and
maintenance tools. `setup.py` collects resources from their canonical source locations.

## Publishing configuration

In the release repository, configure a GitHub environment named `pypi`:

- Add the environment secret `PYPI_API_TOKEN` with an account-wide PyPI token whose account can
  publish all three distributions. This workflow uploads them together; a token scoped to one
  project cannot authorize the other two.
- Allow deployments from `v*` tags.
- Add required reviewers if releases need approval before upload.

See [PyPI token setup](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#create-an-account)
and [GitHub environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

## Publish

1. Merge the release changes, including the selected version and changelog, into `main`.
   Require source CI and the applicable release checks to pass for that commit.
2. Create and push the matching `v<version>` tag in `inclusionAI/ASpec`.
   Mark a corresponding GitHub Release as a prerelease when the package version is a prerelease.
3. In **Actions → Build and publish release**, require `build` and `wheel-install` to pass.
   Review the `aspec-distributions` artifact and approve the `pypi` deployment if approval is configured.
4. The `publish` job verifies checksums and uploads the same wheel and sdist to PyPI.
   It runs only for version-tag pushes in `inclusionAI/ASpec`.
5. Check the [PyPI release](https://pypi.org/project/ling-aspec/) and follow the
   [installation guide](user-guide/installation.md) in a fresh Linux environment.

PyPI does not allow replacing an uploaded distribution file. Inspect an interrupted upload
before retrying; source changes require a new version. Retain the build and validation records.
The workflow does not create a GitHub Release.
