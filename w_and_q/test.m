a=diag([10,8,5]);
b=AU'*a*AU
[c,d]=eig(b);
% AU
% d
% c'
%c*b*c'
% AU*[10,8,5]'-c'*[10,8,5]'
%c'*d*c
AU
c'