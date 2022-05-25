clear all
profile on

addpath('img'); 
addpath('util');

%%%%%%%%%%%%%%%%%%%%%  smooth image  %%%%%%%%%%%%%%
lambda = 25; %        %%%%%  
p1= 2;  % 
p2 = 4;
a = 1;                                                      
b = 15;  


iter = 300;
threshold_res = 2e-5;

uclean = imread('hill.png');                     
        
u0 = imnoise(uclean,'poisson');

u0 = double(u0);
uclean = double(uclean);

[l1, l2] = size(u0);
Area = l1*l2;

lambda11 = u0*0; lambda12 = u0*0; 
lambda2 = u0*0;
x1 = u0*0; x2 = u0*0; 
y = u0*0;
u = u0;

residual = zeros(2,iter);
energy = zeros(1,iter);
error = zeros(2,iter);
%PSNR = zeros(1,iter); SSIM = zeros(1,iter);
relative_error = zeros(1,iter);
DtD = abs(psf2otf([1,-1],[l1, l2])).^2 + abs(psf2otf([1;-1],[l1, l2])).^2;
A = (p1)*DtD + p2;

real_iter = iter;
record_flag = 0;
t1=clock;
for i=1:iter
    
    u_old = u;
    
    g = - dxb(p1*x1 + lambda11) - dyb(p1*x2 + lambda12) + p2*y + lambda2;
    g = fftn(g);
    u = real(ifftn(g./A));
    
    %%%%%%%%%%%%%%%%%%%%%  Update afa
    k = DisCuNew(u);
    afa = a + b*(abs(k));
  
    %%%%%%%%%%%%%%%%  For x
    
    for j = 1:5 % 10
    Dx1 = afa.*(x1./sqrt(1 + x1.^2)) + p1*(x1 - dxf(u) + lambda11/p1);
    D2x1 =  afa.*(1 + x1.^2).^(-1.5) + p1;   
    x1 = x1 - Dx1./D2x1;
    
    Dx2 = afa.*(x2./sqrt(1 + x2.^2)) + p1*(x2 - dyf(u) + lambda12/p1);
    D2x2 =  afa.*(1 + x2.^2).^(-1.5) + p1;   
    x2 = x2 - Dx2./D2x2;
    end 
    
    %%%%%%%%%%%%%%%%  For y
    y1 = (lambda + lambda2)/p2;
    y = (u - y1 + sqrt((y1 - u).^2 + (4*lambda*u0)/p2))/2;
    
    lambda11_old = lambda11;
    lambda12_old = lambda12;
    lambda2_old = lambda2;
    
    lambda11 = lambda11 + p1*(x1 - dxf(u));
    lambda12 = lambda12 + p1*(x2 - dyf(u));
    lambda2 = lambda2 + p2*(y - u);
    
    R11 = x1 - dxf(u);
    R12 = x2 - dyf(u);
    R2 = y - u;
    
    energy(i) = TCEulerPoisson_energy(u,u0,x1,x2,afa,lambda);
    

    %PSNR(i) = psnr(uint8(u),uint8(uclean));
    %SSIM(i) = ssim(uint8(u),uint8(uclean));
    relative_error(i) =  sum(sum( abs(u-u_old) ))/sum(sum(u_old)); 
   % relative_error(i) =  norm(u-u_old,'fro')/Area; 
    
      if( relative_error(i) < threshold_res )
        if( record_flag==0 )
            real_iter = i;
            record_flag = 1;
        end
      end   
end
t2=clock;
t=etime(t2,t1);

fprintf(' The iteration number is: %10d\n', real_iter);
fprintf(' The iteration time is: %4.2fs', t);
fprintf(' The relative error is: %10.8f\n', relative_error(real_iter));

% psnr_u0 = psnr(uint8(u0),uint8(uclean))
% ssim_u0 = ssim(uint8(u0),uint8(uclean))
psnr_u = psnr(uint8(u),uint8(uclean))
ssim_u = ssim(uint8(u),uint8(uclean))

iternum = 1:i;
figure;
plot(iternum,log(relative_error),'r','LineWidth',2); 
xlabel('Iteration')
ylabel('Relative error in u^k')
%title('relative error');

figure;
plot(iternum,log(energy),'b','LineWidth',2);   
xlabel('Iteration')
ylabel('Energy')

% figure;
% plot(iternum,PSNR,'b','LineWidth',2);
% xlabel('Iteration')
% ylabel('PSNR')
% 
% figure;
% plot(iternum,SSIM,'b','LineWidth',2);
% xlabel('Iteration')
% ylabel('SSIM')

figure;
imshow(uint8(u0));
%title('The input noisy image');


figure;
imshow(uint8(u));
%title('The result');


figure;
imshow(uint8(u0-u+100));
%title('The difference');

 profile off
 profile viewer