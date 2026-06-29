function 	Q=calc_Q(x)
global inertia_t d_time
global VarOfProc
B=1./diag(inertia_t).*eye(3);
[phiX,FX]=RecursionFunction(x);
M=FX(4:6,4:6);

QQ=zeros(6,6);
QQ(1:3,1:3)=B*B*(d_time^3)/12;
QQ(1:3,4:6)=B*B*M'/6*((d_time)^3)+B*B/4*((d_time)^2);
QQ(4:6,1:3)=QQ(1:3,4:6)';
%QQ=zeros(6,6);
QQ(4:6,4:6)=M*B*B*M'*(d_time^3)/3+((d_time)^2)/2*(B*B*M'+M*B*B)...
    +B*B*d_time;

Q=2*mean(VarOfProc).*QQ;%%????
% Q=VarOfProc.*QQ;