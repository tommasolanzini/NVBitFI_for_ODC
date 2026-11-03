#!/bin/bash
# See test-apps/simple_add/sdc_check.sh for the meaning of the log files.

# Full output matrices. cmp is silent when the files are equal.
: > diff.log
for m in core tc; do
	cmp output_${m}.bin ${APP_DIR}/golden_output_${m}.bin >> diff.log 2>&1
done

diff stdout.txt ${APP_DIR}/golden_stdout.txt > stdout_diff.log
diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log

# No application-specific check: the matrices are compared bit by bit above
touch special_check.log
