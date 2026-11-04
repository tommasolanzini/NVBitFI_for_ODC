#!/bin/bash
# See test-apps/simple_add/sdc_check.sh for the meaning of the log files.

# Full output. cmp is silent when the files are equal.
cmp output_logits.bin ${APP_DIR}/golden_output_logits.bin > diff.log 2>&1

diff stdout.txt ${APP_DIR}/golden_stdout.txt > stdout_diff.log
diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log

# No application-specific check: the output is compared bit by bit above
touch special_check.log
