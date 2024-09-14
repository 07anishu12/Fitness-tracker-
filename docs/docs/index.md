# health tracker documentation!

## Description

Checks various perameter of the health using smartwatch data

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://my health/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://my health/data/` to `data/`.


