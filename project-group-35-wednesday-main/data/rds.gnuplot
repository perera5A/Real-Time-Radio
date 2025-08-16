set ylabel 'Sample value'               # set y-axis label
set xlabel 'Sample #'                   # set x-axis label
set yrange [-1:1]                       # set y plot range
set xrange [200:600]                      # set x plot range

set multiplot layout 4,1 title "Signal Processing Plots"
#plot "../data/mixed_I.dat" using 1:2 with lines title "mixed I" lw 2, \
#     "../data/mixed_Q.dat" using 1:2 with lines title "mixed Q" lw 2 lc rgb "red"

#plot "../data/lpf_I.dat" using 1:2 with lines title "lpf I" lw 2, \
#     "../data/lpf_Q.dat" using 1:2 with lines title "lpf Q" lw 2 lc rgb "red"

#plot "../data/resamp_I.dat" using 1:2 with lines title "resampled I" lw 2, \
#     "../data/resamp_Q.dat" using 1:2 with lines title "resampled Q" lw 2 lc rgb "red"

# First plot: RRC and Pre-RRC
plot "../data/RRC_data.dat" using 1:2 with lines title "RRC Output" lw 2, \
     "../data/pre_rrc_data.dat" using 1:2 with lines title "Pre-RRC Output" lw 2 lc rgb "red"

# Second plot: I and Q Samples

set yrange [-1:1]                       # set y plot range
set xrange [500:1000]                      # set x plot range
plot "../data/I_sample.dat" using 1:2 with lines title "I sample" lw 2, \
     "../data/Q_sample.dat" using 1:2 with lines title "Q sample" lw 2 lc rgb "red" dt 3

# Constellation plot
set xrange [-1:1]
set yrange [-1:1]
set xlabel "I sample"
set ylabel "Q sample"
plot "../data/samples_rrc.dat" using 1:2 with points title "I vs Q"

# bit stream plot
plot "../data/bit_stream.dat" using 1:2 with lines title "bit stream"

unset multiplot
