function [A]=quat2cosmatrix(q)
     R1=[-1*q(1:3) tilde(q(1:3))+q(4).*eye(3)];
     R2=[-1*q(1:3) -1*tilde(q(1:3))+q(4).*eye(3)];
     R3=R1*R2';
     A=R3;