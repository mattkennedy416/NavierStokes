


xVals = x(:,1);


%% pres heatmap
figure(1)
surf(x/xmax, y/xmax, p/p0); view(2)
colorbar
shading interp
axis([0,1,0,ymax/xmax])

xlabel('x/(plate length) along plate')
ylabel('y/(plate length) away from plate')
title('Pressure Ratio to Freestream, M_\infty = 4.0')


%% temp heatmap

figure(2)
surf(x/xmax, y/xmax, T/T0); view(2)
colorbar
shading interp
axis([0,1,0,ymax/xmax])

xlabel('x/(plate length) along plate')
ylabel('y/(plate length) away from plate')
title('Temperature Ratio to Freestream, M_\infty = 4.0')


%% pressure and temp ratios

figure(3)
plot(xVals/xmax,p(:,1)/p0, 'LineWidth', 2)
hold on
plot(xVals/xmax,T(:,1)/T0, 'LineWidth', 2) % we can actually put these on the same plot

xlabel('x/(plate length) along plate')
ylabel('Ratio from freestream values')
legend('P/P_{\infty}', 'T/T_{\infty}')
title('Pressure and Temperature Along Plate, M_\infty = 4.0')


%% quiver plot
figure(4)
% just plot every other point
quiver(x(1:2:end,1:2:end)/xmax, y(1:2:end,1:2:end)/xmax, u(1:2:end,1:2:end), v(1:2:end,1:2:end))
axis([0,1,0,0.25])

xlabel('x/(plate length) along plate')
ylabel('y/(plate length) away from plate')
title('Velocity Vectors, M_\infty = 4.0')
