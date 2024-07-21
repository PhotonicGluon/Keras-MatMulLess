# Keras-MML Documentation

## Organization

The documentation is organized as follows:

- source files are located in `source`, except for the API pages which will be generated dynamically;
- `release` contains files relating to the *next release of Keras-MML*, like the release notes; and
- the `build` directory (not included in GitHub) contains the actual files displayed on the website.

## Building

To build the documentation while in the root directory, run

```bash
python docs/build.py
```

## Cleaning

To clean up the build (and generated) folders, run

```bash
python clean.py
```
