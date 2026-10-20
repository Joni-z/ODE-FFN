# Working notes

- Torch login currently fails authentication.
- MBP is reachable; its connection to 145 is closed remotely.
- Existing trainer samples against test targets during training. New configurations
  must explicitly select validation sampling; preserve legacy defaults.
- Removing gate modules normally shifts subsequent initialization. Construct and
  discard them before scalar/fixed replacement to preserve every common weight.
- The submitted report is not being revised.
