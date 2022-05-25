function k=DisCuNew(u)

dist = zeros([size(u),8]);

u1 = u([end 1:end-1],:);
u2 = u([2:end 1],:);
u3 = u(:,[end 1:end-1]) ;
u4 = u(:,[2:end 1]);
u5 = u([end 1:end-1],[end 1:end-1]);
u6 = u([2:end 1],[2:end 1]);
u7 = u([end 1:end-1],[2:end 1]);
u8 = u([2:end 1],[end 1:end-1]);

u01 = (u([end 1:end-1],:) + u(:,:))/2;
u02 = (u([2:end 1],:) + u(:,:))/2;
u03 = (u(:,[end 1:end-1]) + u(:,:))/2;
u04 = (u(:,[2:end 1]) + u(:,:))/2;
u05 = (u([end 1:end-1],[end 1:end-1]) + u(:,[end 1:end-1]) + u([end 1:end-1],:) + u(:,:))/4;
u06 = (u([2:end 1],[2:end 1]) + u(:,[2:end 1]) + u([2:end 1],:) + u(:,:))/4;
u07 = (u([end 1:end-1],[2:end 1]) + u(:,[2:end 1]) + u([end 1:end-1],:) + u(:,:))/4;
u08 = (u([2:end 1],[end 1:end-1]) + u(:,[end 1:end-1]) + u([2:end 1],:) + u(:,:))/4;


dist(:,:,1) = ((2*u(:,:)-u3-u4)./sqrt((2*u1-u3-u4).^2 + (u3-u4).^2 + 4))./((u01 - u(:,:)).^2 + 0.25);

dist(:,:,2) = ((u3+u4-2*u(:,:))./sqrt((2*u2-u3-u4).^2 + (u4-u3).^2 + 4))./((u02 - u(:,:)).^2 + 0.25);

dist(:,:,3) = ((u1+u2-2*u(:,:))./sqrt((u1+u2-2*u3).^2 + (u2-u1).^2 + 4))./((u03 - u(:,:)).^2 + 0.25);

dist(:,:,4) = ((2*u(:,:)-u1-u2)./sqrt((u1+u2-2*u4).^2 + (u1-u2).^2 + 4))./((u04 - u(:,:)).^2 + 0.25);

dist(:,:,5) = ((u5+u7+u8-u1-u3-u(:,:))./sqrt((u8-u5).^2 + (u7-u5).^2 + 4))./((u05 - u(:,:)).^2 + 0.5);

dist(:,:,6) = ((u(:,:)+u2+u4-u6-u7-u8)./sqrt((u7-u6).^2 + (u8-u6).^2 + 4))./((u06 - u(:,:)).^2 + 0.5);

dist(:,:,7) = ((u(:,:)+u1+u4-u5-u6-u7)./sqrt((u7-u6).^2 + (u5-u7).^2 + 4))./((u07 - u(:,:)).^2 + 0.5);

dist(:,:,8) = ((u5+u6+u8-u2-u3-u(:,:))./sqrt((u8-u5).^2 + (u6-u8).^2 + 4))./((u08 - u(:,:)).^2 + 0.5);


[row,col] = ndgrid(1:size(u,1),1:size(u,2)); row=uint32(row); col=uint32(col);
tmp = abs(dist); 
[~,ind1] = min(tmp,[],3); %寻找8个矩阵中每一个位置的最小值 127*127 [1-8]
[~,ind2] = max(tmp,[],3);
dim1 = uint32(size(dist,1)); %127 
dim2 = uint32(size(dist,1)*size(dist,2)); %127*127
index1 = row + dim1.*uint32(col-1) + dim2.*uint32(ind1-1);
index2 = row + dim1.*uint32(col-1) + dim2.*uint32(ind2-1);
k = dist(index1).*dist(index2);
%k = (dist(index1) + dist(index2))/2;