reset
set ylabel 'Operations Count'
set style fill solid
set title 'operation count comparison'
set term png enhanced font 'Verdana,10'
set output 'operation_comparison.png'

plot [:][0:]'operation_count.txt' using 2:xtic(1) with histogram title 'Shift' ,\
        '' using 3:xtic(1) with histogram title 'Detect' ,\
        '' using 4:xtic(1) with histogram title 'Insert' ,\
        '' using 5:xtic(1) with histogram title 'Delete' ,\
        '' using ($0):($2):2 with labels title ' ' ,\
        '' using ($0):($3):3 with labels title ' ' ,\
        '' using ($0):($4):4 with labels title ' ' 
