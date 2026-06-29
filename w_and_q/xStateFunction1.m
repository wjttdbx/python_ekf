function XNew=xStateFunction1(X,w)
global d_time inertia_t
qa=X;
qa=qa+(QuanMatrix(w))*qa*d_time/2;
q=(quatnormalize(qa'))';
XNew=q;
