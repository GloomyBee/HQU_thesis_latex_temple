function myContourf(x,y,z,scale)
%Used in visualization
    contourf(x,y,z,scale,'LineStyle','none');
    set(gca,'ticklabelinterpreter','latex','fontsize',11)
    colormap(jet);
    colorbar('ticklabelinterpreter','latex')
end