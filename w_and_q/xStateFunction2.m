function XNew=xStateFunction2(X,T)
global d_time inertia_t
w=X;
wd=inertia_t\( T-cross( w, (inertia_t)*w ) );
w=w+ d_time * wd;
XNew=w;