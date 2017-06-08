import numpy as np
import random
import collections


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # All angles visible to the rover. normalized by 5: degree_options
        degree_options = np.round(Rover.nav_angles * 180 / (5 * np.pi)) * 5

        # Filter only angle forward: [-15 -> 15]
        forward_angle = degree_options[degree_options <= 15.0]
        forward_angle = forward_angle[forward_angle >= -15]


        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(forward_angle) >= Rover.go_forward:

                # at high speed, turn slowly
                slow_turn = 0.5 if Rover.vel > 1 else 1

                angle_options = collections.Counter(forward_angle)
                picked_angle = angle_options.most_common(1)[0][0]
                print("Keep moving forward with angle: ", picked_angle)
                Rover.steer = picked_angle * slow_turn

                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            else:
                print("Can't move forward... Stopping")
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                print("Breaking...")
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(forward_angle) > Rover.go_forward: # need more space to start moving
                    angle_options = collections.Counter(forward_angle)
                    picked_angle = angle_options.most_common(1)[0][0]

                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to picked angle
                    Rover.steer = picked_angle
                    Rover.mode = 'forward'
                    print("Start moving forward with angle: ", picked_angle)

                else:
                    # if cannot move forward, keep steering to the same direction
                    if Rover.steer:
                        picked_angle = Rover.steer
                    # if no forward angle, look for any path
                    elif len(degree_options) > Rover.go_forward:
                        angle_options = collections.Counter(degree_options)
                        sign = 1 if angle_options.most_common(1)[0][0] >= 0 else -1
                        # move 5 degrees towards angle options (positive or negative direction)
                        picked_angle = sign * 5
                    else:
                        picked_angle = 5

                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    Rover.steer = picked_angle
                    print("UTurn with angle: ", picked_angle)

    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover

