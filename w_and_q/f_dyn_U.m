function [vdU,wdU]=f_dyn_U(RU,R_touch_e,AU,vU,wU,F,T)
global mt inertia_t
vdU=F/mt;
T_ex=T+cross( (R_touch_e-RU),F);
%wdU=inv(AU*inertia_t*AU')*( T_ex-cross( wU, (AU*inertia_t*AU')*wU ) );
wdU=inertia_t\( T_ex-cross( wU, (inertia_t)*wU ) );