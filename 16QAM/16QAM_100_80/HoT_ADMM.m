function x = HoT_ADMM(HTH, HTy, Lf, x)

[~, N] = size(HTH);

u = zeros(N, 1);

L_bar = Lf;

max_iter = 100;
gamma = 1/3;

x_pre = x;
x_pre_out = x;
for ii = 1:10
    z = x; 
    for jj = 1:max_iter

        x = inv(HTH+L_bar*eye(N))*(HTy+u+L_bar*z);

        temp = x-u/L_bar;

        z = tanh(temp/gamma*0.5) + tanh((temp+2)/gamma*0.5) + tanh((temp-2)/gamma*0.5);

        u = u+L_bar*(z-x);

        if norm(x_pre-x)/N<1e-2 && norm(x-z)/N<1e-2
            break
        end
        x_pre = x;
    end
    gamma = 0.5*gamma;

    if norm(x_pre_out-x)/N<1e-4
        break
    end
    x_pre_out = x;
end
