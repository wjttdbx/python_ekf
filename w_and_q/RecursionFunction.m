function [phiX,FX]=RecursionFunction(xForm)
global inertia_t d_time
%q在前，w在后
%共六维
qForm=xForm(1:3);
wForm=xForm(4:6);
%AForm=quat2cosmatrix(qForm);
p1=(inertia_t(2,2)-inertia_t(3,3))/inertia_t(1,1);
p2=(inertia_t(3,3)-inertia_t(1,1))/inertia_t(2,2);
p3=(inertia_t(1,1)-inertia_t(2,2))/inertia_t(3,3);

FX=zeros(6,6);
FX(1:3,1:3)=-1.*tilde(wForm);
FX(1:3,4:6)=eye(3).*0.5;

FX(4:6,4:6)=[0 -1*p1 p1;p2 0 -1*p2;-1*p3 p3 0].*(tilde(wForm));
phiX=eye(6)+d_time.*FX;