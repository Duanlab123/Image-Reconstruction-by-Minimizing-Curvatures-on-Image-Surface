function energy = TCEulerPoisson_energy(u,u0,x1,x2,afa,lambda)

%%%%%%%%%%%%%%%%%%%%%  Energy
norm_x = sqrt(x1.^2 + x2.^2 + 1);
normx = afa.*norm_x;

energy = sum(sum(u-u0.*log(u))) + lambda*sum(normx(:));