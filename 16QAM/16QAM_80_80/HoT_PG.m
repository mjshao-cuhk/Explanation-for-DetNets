function x = HoT_PG(x_ini,HH,Hy,L, cons)

N = length(x_ini);
in_iter=200;
x=x_ini;

s = length(cons);
%cons_ex = repmat(cons,N,1);

alpha = 1.5/L;

for i_mu=1:10
    y_x=x;
    t=1;
    for i=1:in_iter

        grad_f=2*(HH*y_x-Hy);

        x_buff = L*y_x-grad_f;

        temp1 = tanh(alpha*(x_buff+L*2));
        temp2 = tanh(alpha*(x_buff-L*2));
        temp3 = tanh(alpha*x_buff);

        ply_x = temp1 + temp2 + temp3;

        x_pre=x;
        x = ply_x;
        t1 = (1+sqrt(1+4*t^2))/2;
        y_x = x + (t-1)/t1 *(x-x_pre);
        t=t1;

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
    if abs(alpha_pre-alpha)<1e-4
        break
    end

end