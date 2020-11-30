% function Test_group_lasso

% min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

% generate data
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
n = 512;
m = 256;
A = randn(m,n);
k = round(n*0.1); l = 2;
A = randn(m,n);
p = randperm(n); p = p(1:k);
u = zeros(n,l);  u(p,:) = randn(k,l);  
b = A*u;
mu = 1e-2;

fig = figure(1);
subplot(2,1,1); 
% plot(1:n, u(:,1), '*');
plot(1:n, u(:,1), '*', 1:n, u(:,2), 'o'); 
xlim([1 n])
title('(1) exact solution $u$', 'Interpreter','latex');

x0 = randn(n, l);

errfun = @(x1, x2) norm(x1 - x2, 'fro') / (1 + norm(x1,'fro'));
errfun_exact = @(x) norm(x - u, 'fro') / (1 + norm(u,'fro'));
sparisity = @(x) sum(abs(x(:)) > 1E-6 * max(abs(x(:)))) /(n*l);

% cvx calling mosek
opts1 = []; % modify options
tic;
[x1, iter1, out1] = gl_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

% cvx calling gurobi
opts2 = []; % modify options
tic;
[x2, iter2, out2] = gl_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;

% call mosek directly
opts3 = []; % modify options
tic;
[x3, iter3, out3] = gl_mosek(x0, A, b, mu, opts3);
t3 = toc;

% call gurobi directly
opts4 = []; % modify options
tic;
[x4, iter4, out4] = gl_gurobi(x0, A, b, mu, opts4);
t4 = toc;

% other approaches
% Subgradient Method
opts5 = []; % modify options
tic;
[x5, iter5, out5] = gl_SGD_primal(x0, A, b, mu, opts5);
t5 = toc;

% Gradient Method for the Smoothed Primal Problem
opts6 = []; % modify options
tic;
[x6, iter6, out6] = gl_GD_primal(x0, A, b, mu, opts6);
t6 = toc;

% Fast (Nesterov/Accelerated) Gradient Method for the Smoothed Primal Problem.
opts7 = []; % modify options
tic;
[x7, iter7, out7] = gl_FGD_primal(x0, A, b, mu, opts7);
t7 = toc;

% Proximal Gradient Method for the Primal Problem
opts8 = []; % modify options
tic;
[x8, iter8, out8] = gl_ProxGD_primal(x0, A, b, mu, opts8);
t8 = toc;

% Fast Proximal Gradient Method for the Primal Problem
opts9 = []; % modify options
tic;
[x9, iter9, out9] = gl_FProxGD_primal(x0, A, b, mu, opts9);
t9 = toc;

% Augmented Lagrangian Method for the Dual Problem
opts10 = []; % modify options
tic;
[x10, iter10, out10] = gl_ALM_dual(x0, A, b, mu, opts10);
t10 = toc;

% Alternating Direction Method of Multipliers for the Dual Problem
opts11 = []; % modify options
tic;
[x11, iter11, out11] = gl_ADMM_dual(x0, A, b, mu, opts11);
t11 = toc;

% Alternating Direction Method of Multipliers for the Primal Problem
opts12 = []; % modify options
tic;
[x12, iter12, out12] = gl_ADMM_primal(x0, A, b, mu, opts12);
t12 = toc;

% Proximal Point Method for the Dual Problem
opts13 = []; % modify options
tic;
[x13, iter13, out13] = gl_PPA_dual(x0, A, b, mu, opts13);
t13 = toc;

% Block Coordinate Method for the Primal Problem
opts14 = []; % modify options
tic;
[x14, iter14, out14] = gl_BCD_primal(x0, A, b, mu, opts14);
t14 = toc;


%% print comparison results with cvx-call-mosek
fprintf('     CVX-Mosek: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t1, iter1, out1.fval, sparisity(x1), errfun_exact(x1), errfun(x1, x1), errfun(x2, x1));
fprintf('    CVX-Gurobi: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t2, iter2, out2.fval, sparisity(x2), errfun_exact(x2), errfun(x1, x2), errfun(x2, x2));
fprintf('         Mosek: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t3, iter3, out3.fval, sparisity(x3), errfun_exact(x3), errfun(x1, x3), errfun(x2, x3));
fprintf('        Gurobi: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t4, iter4, out4.fval, sparisity(x4), errfun_exact(x4), errfun(x1, x4), errfun(x2, x4));
fprintf('    SGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t5, iter5, out5.fval, sparisity(x5), errfun_exact(x5), errfun(x1, x5), errfun(x2, x5));
fprintf('     GD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t6, iter6, out6.fval, sparisity(x6), errfun_exact(x6), errfun(x1, x6), errfun(x2, x6));
fprintf('    FGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t7, iter7, out7.fval, sparisity(x7), errfun_exact(x7), errfun(x1, x7), errfun(x2, x7));
fprintf(' ProxGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t8, iter8, out8.fval, sparisity(x8), errfun_exact(x8), errfun(x1, x8), errfun(x2, x8));
fprintf('FProxGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t9, iter9, out9.fval, sparisity(x9), errfun_exact(x9), errfun(x1, x9), errfun(x2, x9));
fprintf('      ALM Dual: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter10, out10.fval, sparisity(x10), errfun_exact(x10), errfun(x1, x10), errfun(x2, x10));
fprintf('     ADMM Dual: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t11, iter11, out11.fval, sparisity(x11), errfun_exact(x11), errfun(x1, x11), errfun(x2, x11));
fprintf('   ADMM Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t12, iter12, out12.fval, sparisity(x12), errfun_exact(x12), errfun(x1, x12), errfun(x2, x12));
fprintf('      PPA dual: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t13, iter13, out13.fval, sparisity(x13), errfun_exact(x13), errfun(x1, x13), errfun(x2, x13));
fprintf('    BCD primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t14, iter14, out14.fval, sparisity(x14), errfun_exact(x14), errfun(x1, x14), errfun(x2, x14));

plot_results(u, 'Exact', '../figures/gl_exact.png', u, x1, x2)
plot_results(x1, 'CVX-Mosek', '../figures/gl_cvx_mosek.png', u, x1, x2)
plot_results(x2, 'CVX-Gurobi', '../figures/gl_cvx_gurobi.png', u, x1, x2)

plot_results(x3, 'Mosek', '../figures/gl_mosek.png', u, x1, x2)
plot_results(x4, 'Gurobi', '../figures/gl_gurobi.png', u, x1, x2)

plot_results(x5, 'SGD Primal', '../figures/gl_SGD_Primal.png', u, x1, x2)
plot_results(x6, 'GD Primal', '../figures/gl_GD_primal.png', u, x1, x2)

plot_results(x7, 'FGD Primal', '../figures/gl_FGD_primal.png', u, x1, x2)
plot_results(x8, 'ProxGD Primal', '../figures/gl_ProxGD_primal.png', u, x1, x2)
plot_results(x9, 'FProxGD Primal', '../figures/gl_FProxGD_primal.png', u, x1, x2)

plot_results(x10, 'ALM Dual', '../figures/gl_ALM_dual.png', u, x1, x2)
plot_results(x11, 'ADMM Dual', '../figures/gl_ADMM_dual.png', u, x1, x2)
plot_results(x12, 'ADMM Primal', '../figures/gl_ADMM_primal.png', u, x1, x2)

plot_results(x13, 'PPA Dual', '../figures/gl_PPA_dual.png', u, x1, x2)
plot_results(x14, 'BCD Prlmal;', '../figures/BCD_primal.png', u, x1, x2)

