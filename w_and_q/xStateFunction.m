function XNew=xStateFunction(X,T)
global d_time inertia_t
qa=X(1:4);
w=X(5:7);
wd=inertia_t\( T-cross( w, (inertia_t)*w ) );
qa=qa+(QuanMatrix(w))*qa*d_time/2;
w=w+ d_time * wd;
q=(quatnormalize(qa'))';
XNew(1:4)=q;
XNew(5:7)=w;