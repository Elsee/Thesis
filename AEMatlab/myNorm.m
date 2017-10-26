function normMatr = myNorm(mat, dim)
    normMatr = mat;
    if dim == 2
        for i = 1:size(mat,2)
            normMatr(:,i) = (mat(:,i)-min(mat(:,i)))/(max(mat(:,i))-min(mat(:,i)));
        end
    else
        for i = 1:size(mat,1)
            normMatr(i,:) = (mat(i,:)-min(mat(i,:)))/(max(mat(i,:))-min(mat(i,:)));
        end
end