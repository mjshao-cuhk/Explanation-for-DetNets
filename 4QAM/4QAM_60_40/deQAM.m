function xout = deQAM(x,cons)

l = length(cons);
n = length(x);
cons_rep = repmat(cons, n, 1);
x_rep = repmat(x,1,l);
temp = abs(x_rep-cons_rep);
[value,index] = min( temp.');
xout = zeros(n,1);
for ii = 1:n
    xout(ii)=cons(index(ii)); 
end