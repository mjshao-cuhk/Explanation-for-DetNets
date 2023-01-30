function x=HoT_PG(x_ini,HH,Hy,L)

N=size(x_ini,1)/2;

in_iter=200;
x=x_ini;
x_out_pre = x*0;

alpha = 0.5;
lambda = 0.5*L;
gamma = L;

for i_mu=1:500
    x_pre = x;
    for i=1:in_iter
 
        grad_f=2*(HH*x-Hy);

        x_buff = lambda*x-grad_f;
        x = 2./(1+exp(-2*x_buff/gamma)) -1 ;
        
        if norm(x-x_pre,'fro')<1e-4
            break
        end
        x_pre=x;
    end
    if norm(x_out_pre-x,'fro')<1e-4
        break
    end
    x_out_pre = x;

    gamma = gamma - alpha*(2*N -norm(x)^2)*i_mu*2;

end
