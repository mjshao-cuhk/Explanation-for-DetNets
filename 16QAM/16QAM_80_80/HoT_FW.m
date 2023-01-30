function x=HoT_FW(x_ini,HH,Hy, H, y,L,cons)

N=size(x_ini,1)/2;
in_iter=100;
x=x_ini;

alpha=0.5;
gamma = 3;

s = length(cons);
%cons_ex = repmat(cons,N,1);

setD = [-2, 0, 2];
partit = [-1,1];
func_val =  @(x,c,gamma) norm(H*x - y)^2 -L/2*norm(x- c)^2 + L/gamma*sum((1+(x-c)).*log(1+(x-c)) +(1-(x-c)).*log(1-(x-c)));

eta = 0:0.01:1;
fun_temp = zeros(length(eta),1);
for i_mu=1:10
    for i=1:in_iter
        grad_f=2*(HH*x-Hy);


        x_buff=x-(1/L)*grad_f;
        c_ind = quantiz(x_buff, partit,setD );
        c=setD(c_ind+1).';
        ply_x = 2./(1+exp(-gamma*(x_buff-c))) -1 + c;
        for j =1 : length(eta)
            x_temp  = x + eta(j)*(ply_x - x);
            c_ind_temp = quantiz(x_temp, partit,setD );
            c_temp=setD(c_ind_temp+1).';
            fun_temp(j) = func_val(x_temp,c_temp, gamma);
        end
        [~,min_ind] = min(fun_temp);
        eta_min = eta(min_ind);
        x_pre = x;
        x = x+ eta_min* (ply_x - x);
        if norm(x-x_pre,'fro')<1e-3
            break
        end
    end
    alpha_pre = alpha;
    %x_temp = repmat(x,1,s);
    temp = abs(x-cons);
    [value,index] = min( temp.');
    cg = abs(cons(index)).';

    alpha = alpha + sum(((cg-value.')./cg).^2)/N;
    if abs(alpha_pre-alpha)<1e-5
        break
    end
end


