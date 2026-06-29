function [xDot,q0chang]=Deriv(xGuess,qGuess,Toe)
global inertia_t 
xDot=zeros(6,1);
qvGuess=xGuess(1:3);
wGuess=xGuess(4:6);
xDot(1:4)=0.5.*QuanMatrix(wGuess)*qGuess;%临时占用第四个元素
q0chang=xDot(4);
xDot(4:6)=inertia_t\( Toe-1*cross( wGuess, (inertia_t)*wGuess ) );