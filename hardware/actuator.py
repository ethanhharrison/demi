import RPi.GPIO as GPIO
import time

SERVO_PIN = 17

def setup_servo() -> GPIO.PWM:
    """
    Initialize the GPIO pin for servo control.

    Sets up the Raspberry Pi GPIO mode, configures the servo signal pin as output,
    and starts a 50 Hz PWM signal (standard for hobby servos).

    Returns:
        GPIO.PWM: A PWM object used to control the servo.
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(0)
    return pwm

def angle_to_duty_cycle(angle: float) -> float:
    """
    Convert a servo angle in degrees (0–180) to the corresponding PWM duty cycle.

    Args:
        angle (float): Desired servo angle in degrees.

    Returns:
        float: PWM duty cycle value between approximately 2 and 12.
    """
    return 2 + (angle / 18.0)

def move_servo(n_degrees: float) -> None:
    """
    Move a microservo connected to GPIO 17 by +n degrees, hold for 1 second,
    then return -n degrees back to the neutral position (90°).

    Args:
        n_degrees (float): The number of degrees to rotate from the neutral position.

    Behavior:
        - Starts PWM on GPIO 17.
        - Moves servo to neutral + n degrees.
        - Holds for 1 second.
        - Moves servo to neutral - n degrees.
        - Returns servo to neutral.
        - Cleans up GPIO and stops PWM after completion.
    """
    pwm = setup_servo()

    try:
        neutral = 90
        target_angle = neutral + n_degrees
        return_angle = neutral - n_degrees
        pwm.ChangeDutyCycle(angle_to_duty_cycle(target_angle))
        time.sleep(0.5)
        time.sleep(1.0)
        pwm.ChangeDutyCycle(angle_to_duty_cycle(return_angle))
        time.sleep(0.5)
        pwm.ChangeDutyCycle(angle_to_duty_cycle(neutral))
        time.sleep(0.5)
        pwm.ChangeDutyCycle(0)

    finally:
        pwm.stop()
        GPIO.cleanup()