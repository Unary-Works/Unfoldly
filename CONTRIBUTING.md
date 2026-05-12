# Contributing to Unfoldly

Thanks for your interest in contributing to Unfoldly.

Unfoldly is an early-stage, local-first desktop app for AI-powered file search. We welcome focused bug fixes, documentation improvements, performance work, parser improvements, and carefully scoped product contributions.

## Maintainer Review

All pull requests must be reviewed and approved by Unary Works maintainers before they are merged.

We appreciate every contribution, but submitting a pull request does not guarantee that it will be merged. Once a change is merged, we become responsible for maintaining it for current and future users, so we may ask for changes, suggest a different direction, or close a pull request that does not fit the project.

We review contributions with the long-term product experience in mind, so we may sometimes say no to changes that are useful but not aligned with the current direction.

## Good First Contributions

Good first contributions include:

- fixing typos or unclear documentation
- improving setup instructions
- adding small parser improvements
- improving error messages
- fixing reproducible bugs
- adding synthetic test fixtures
- improving accessibility issues in existing UI

If you are new to the project, these are the best places to start.

## Before You Start

Please open an issue before working on non-trivial changes, especially changes involving:

- user-facing UI or UX
- indexing behavior
- retrieval quality
- model loading or model selection
- file parsing
- image, audio, or video processing
- privacy-sensitive data handling
- app packaging
- architecture changes

Small documentation fixes, typo fixes, and clearly scoped bug fixes can usually go straight to a pull request.

If you are unsure whether a change needs an issue, open one first. We would rather discuss the direction early than close a pull request after you have spent time on it.

## UI Contributions

UI contributions are welcome when they improve clarity, usability, accessibility, responsiveness, or the core file-search workflow.

Please open an issue before starting larger UI changes, including:

- redesigned screens
- new navigation patterns
- new onboarding flows
- major layout changes
- new settings or model-management flows
- visual changes that affect the product identity

For UI pull requests, please include screenshots or short recordings showing the before and after behavior.

Avoid broad visual redesigns without prior maintainer agreement.

## Pull Request Guidelines

For the best chance of review and merge:

1. Keep each pull request focused on one problem.
2. Explain what changed and why.
3. Include screenshots or recordings for UI changes.
4. Include manual testing notes.
5. Add or update tests when practical.
6. Avoid unrelated refactors.
7. Mark work-in-progress pull requests as draft.
8. Do a self-review before requesting review.

A good pull request description includes:

- a short summary of what changed
- why the change is needed
- what user-facing behavior changed, if any
- what manual flows were checked
- known limitations or follow-up work

## Development Setup

The public source tree is still being prepared.

Once the source code is published, this document will be updated with the official development setup, build guidance, and testing workflow.

Until then, please use GitHub issues to discuss proposed changes before opening implementation pull requests.

## Security and Privacy Issues

If you discover a security or privacy issue, please do not open a public issue with sensitive details.

Instead, contact the maintainers directly so we can review the issue before it is disclosed publicly.

## Privacy and Data Safety

Unfoldly works with local personal files. Please be careful with data handling.

Do not commit:

- private files
- user-derived datasets
- local indexes
- downloaded model files
- generated embeddings
- logs
- crash reports with private paths
- benchmark outputs containing personal data
- environment files
- local credentials

Use synthetic fixtures for tests and examples.

## Model, Retrieval, and Indexing Changes

Changes to models, embeddings, reranking, query routing, indexing, OCR, transcription, or file parsing can affect search quality in subtle ways.

For these pull requests, please include:

- the motivation for the change
- example queries or files used for testing
- before and after behavior when relevant
- any tradeoffs in speed, memory, accuracy, or file-type coverage

Avoid hardcoding behavior for one private dataset, one local path, or one personal workflow.

## Documentation

Documentation changes are welcome.

Please keep documentation:

- accurate to the current codebase
- clear about beta limitations
- free of private paths or internal-only references
- easy for new users and contributors to follow

## License

By contributing to Unfoldly, you agree that your contributions will be licensed under the Apache License 2.0.
