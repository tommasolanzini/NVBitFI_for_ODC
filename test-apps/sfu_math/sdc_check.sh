#!/bin/bash
# See test-apps/simple_add/sdc_check.sh for the meaning of the log files.

# Full outputs. cmp is silent when the files are equal.
: > diff.log
for f in rsqrt rcp sin exp log; do
	cmp output_${f}.bin ${APP_DIR}/golden_output_${f}.bin >> diff.log 2>&1
done

diff stdout.txt ${APP_DIR}/golden_stdout.txt > stdout_diff.log
diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log

# No application-specific check: the outputs are compared bit by bit above
touch special_check.log
