for x in `ls *.jsonl`; do echo $x; echo `tail -n2 $x`; echo "scale=2;`grep "true" $x|wc -l`/$(tail -n2 $x|head -n1)"|bc; done
