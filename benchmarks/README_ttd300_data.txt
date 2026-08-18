ttd300 -- TTP-D drone-endurance benchmark (relative endurance)
sizes (customers) : [10, 20, 30, 40, 50, 75, 100]
layouts per size  : [1, 2, 3, 4, 5]
endurance         : ED = round(frac * d_max), frac in [0.25, 0.5, 0.75, 1.0]
                    d_max = instance's own max CEIL_2D pairwise distance
box               : uniform integer coords on [0,300]^2, CEIL_2D
items             : 5/customer, w~U[1000,1009], p~U[1,1000]
capacity          : fixed per size W = round(2275.0357 * n) (a280 per-node ratio 637010/280); same W shared by L1..L5
speeds            : v_min=0.1, v_max=1 (drone = 2x v_max downstream)
renting ratio R   : 50.0
filename _f025/_f050/_f075/_f100 = frac; absolute ED is in the header.
the ...L<L>.txt file (no header) is the unbounded reference.
