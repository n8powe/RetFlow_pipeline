function [camPos] = imu2position(imuData)
    
    sampleRate = 1/220;

    %imuData = imuData(imuData.index==i,:);
    %imuData = mean(imuData(:,4:end), 1);

    velocity.X = cumtrapz(sampleRate,imuData.accelerationX_G_ );
    velocity.Y = cumtrapz(sampleRate,imuData.accelerationY_G_ );
    velocity.Z = cumtrapz(sampleRate,imuData.accelerationZ_G_ );
    x = cumtrapz(sampleRate,velocity.X);
    y = cumtrapz(sampleRate,velocity.Y);
    z = cumtrapz(sampleRate,velocity.Z);

    index = imuData.index;
    camPos = table(index, x, y, z);


end