function x = Box_rel(x_ini, HH, Hy, L, cons)
P_per = ones(length(x_ini), 1)*max(abs(cons));

x=x_ini;
y_x=x;
t=1;
for i_mu=1:200

    grad_f=2*(HH*y_x-Hy);

    x_buff=y_x-(1/L)*grad_f;
    ply_x=min(P_per,max(-P_per,x_buff));

    x_pre=x;
    x=ply_x;
    t1=(1+sqrt(1+4*t^2))/2;
    y_x=x+ (t-1)/t1 *(x-x_pre);
    t=t1;
    if norm(x-x_pre,'fro')<1e-4
        break
    end
end