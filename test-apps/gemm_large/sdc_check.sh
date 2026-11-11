#!/bin/bash
# See test-apps/simple_add/sdc_check.sh for the meaning of the log files.
# Golden of the binary that run.sh ran: golden-<binary>/
G=${APP_DIR}/golden-$(cat gemm_bin.txt)

# Full output matrices. cmp is silent when the files are equal.
: > diff.log
for m in core tc; do
	cmp output_${m}.bin ${G}/golden_output_${m}.bin >> diff.log 2>&1
done

diff stdout.txt ${G}/golden_stdout.txt > stdout_diff.log
diff stderr.txt ${G}/golden_stderr.txt > stderr_diff.log

# No application-specific check: the matrices are compared bit by bit above
touch special_check.log
