function x=HoT_FW(x_ini,HH,Hy, H, y,L)

N=size(x_ini,1)/2;
in_iter=100;
x=x_ini;
x_out_pre = x*0;

gamma = 3;
func_val =  @(x,gamma) norm(H*x - y)^2 -L/2*norm(x)^2 + L/gamma*sum((1+x).*log(1+x) +(1-x).*log(1-x));
eta = 0:0.01:1;
fun_temp = zeros(length(eta),1);
for i_mu=1:8
    for i=1:in_iter
        grad_f=2*(HH*x-Hy);
        x_buff=x-(1/L)*grad_f;
        ply_x = 2./(1+exp(-gamma*x_buff)) -1 ;
        for j =1 : length(eta)
            x_temp  = x + eta(j)*(ply_x - x);
            fun_temp(j) = func_val(x_temp, gamma);
        end
        [~,min_ind] = min(fun_temp);
        eta_min = eta(min_ind);
        x_pre = x;
        x = x+ eta_min* (ply_x - x);
        if norm(x-x_pre,'fro')<1e-3
            break
        end
    end
    if norm(x_out_pre-x,'fro')<1e-4
        break
    end
    x_out_pre = x;
    alpha = 0.5;
    gamma = gamma - alpha*(2*N -norm(x)^2)*i_mu*2;
end


