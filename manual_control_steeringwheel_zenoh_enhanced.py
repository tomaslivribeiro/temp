#!/usr/bin/env python3

"""
X-Verse Virtual Vehicle Module

A CARLA-based vehicle controller with Zenoh integration for the X-Verse architecture.
Handles input from steering wheel and keyboard, publishes vehicle status, and applies
control commands received from other modules.

Controls:
- Steering Wheel (Logitech G920) or Keyboard (WASD/Arrow keys)
- Cruise Control Toggle: Wheel button or 'C' key (only sends request)
- Reverse Toggle: Gear shifter paddles or 'Q' key (only sends request)
- Delta Speed: Wheel buttons or 'Z' key decrease and 'X' key increase (only sends request)
- Camera: TAB to toggle
- Help: H or ?
"""

import os
import sys
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('pygame package required')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('numpy package required')

try:
    import zenoh
except ImportError:
    raise RuntimeError('zenoh package required')

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import RawConfigParser as ConfigParser

# ==============================================================================
# -- Log Tracing control -------------------------------------------------------
# ==============================================================================
# Set this to True if log tracing is intended
_tracing = False

# ==============================================================================
# -- CARLA Import (Conditional) ------------------------------------------------
# ==============================================================================

# CARLA will be imported conditionally in main() based on --test-mode flag
# This allows the script to run on systems without CARLA installed
carla = None
cc = None

def setup_carla_import(test_mode: bool = False):
    """
    Conditionally import CARLA based on test mode
    
    Args:
        test_mode: If True, skip CARLA import and use mock
    
    Returns:
        tuple: (carla_module, color_converter) or (None, None) if mock should be used
    """
    if test_mode:
        logging.info("Test mode: Skipping CARLA import, will use mock implementation")
        return None, None
    
    # Try to add CARLA egg to path
    try:
        import glob
        carla_egg = glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
        sys.path.append(carla_egg)
        logging.debug(f"Added CARLA egg to path: {carla_egg}")
    except IndexError:
        logging.warning("CARLA egg file not found in ../carla/dist/")
    except Exception as e:
        logging.warning(f"Error adding CARLA to path: {e}")
    
    # Try to import CARLA
    try:
        import carla as carla_module
        from carla import ColorConverter as cc_module
        logging.info("CARLA imported successfully")
        return carla_module, cc_module
    except ImportError as e:
        logging.warning(f"CARLA module not found: {e}")
        logging.warning("Falling back to mock implementation")
        return None, None


# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

FIXED_DELTA_SECONDS = 0.05
THROTTLE_DAMPING_FACTOR = 1.75

MIN_THROTTLE = 0.0
MAX_THROTTLE = 1.0
MIN_STEERING = -1.0
MID_STEERING = 0.0
MAX_STEERING = 1.0
MIN_BRAKING = 0.0
MAX_BRAKING = 1.0

RADAR_RANGE_METERS = 60.0
RADAR_HORIZONTAL_FOV_DEG = 20.0
RADAR_VERTICAL_FOV_DEG = 10.0
RADAR_SENSOR_TICK = 0.05
RADAR_POINTS_PER_SECOND = 1500
RADAR_CENTER_AZIMUTH_DEG = 6.0
RADAR_CENTER_ALTITUDE_DEG = 4.0

TRAFFIC_NUM_VEHICLES = 50

# ==============================================================================
# -- SignalDefinitions ---------------------------------------------------------
# ==============================================================================

class SignalDefinitions:
    """Defines all signals used by the Virtual Vehicle Module"""
    
    # Class variables to store signal definitions
    SIGNALS = {}
    PROVIDED_SIGNALS = []
    CONSUMED_SIGNALS = []
    
    @classmethod
    def load_from_file(cls, filepath='signals_config.json'):
        """Load signal definitions from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                
            cls.SIGNALS = config.get('signals', {})
            cls.PROVIDED_SIGNALS = config.get('provided_signals', [])
            cls.CONSUMED_SIGNALS = config.get('consumed_signals', [])
            
            logging.info(f"Loaded {len(cls.SIGNALS)} signal definitions from {filepath}")
            logging.info(f"Provided signals: {len(cls.PROVIDED_SIGNALS)}")
            logging.info(f"Consumed signals: {len(cls.CONSUMED_SIGNALS)}")
            
            # Validate that all referenced signals exist
            for signal in cls.PROVIDED_SIGNALS + cls.CONSUMED_SIGNALS:
                if signal not in cls.SIGNALS:
                    logging.warning(f"Signal '{signal}' referenced but not defined in config")
            
            return True
        except FileNotFoundError:
            logging.error(f"Signal configuration file not found: {filepath}")
            return False
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing signal configuration file: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error loading signal configuration: {e}")
            return False
    
    @classmethod
    def get_topic(cls, signal_name: str) -> str:
        """Get the Zenoh topic for a signal"""
        if signal_name not in cls.SIGNALS:
            raise ValueError(f"Unknown signal: {signal_name}")
        return cls.SIGNALS[signal_name]['topic']
    
    @classmethod
    def get_provided_signals(cls) -> List[str]:
        """Get list of signals provided by Virtual Vehicle"""
        return cls.PROVIDED_SIGNALS
    
    @classmethod
    def get_consumed_signals(cls) -> List[str]:
        """Get list of signals consumed by Virtual Vehicle"""
        return cls.CONSUMED_SIGNALS


# ==============================================================================
# -- MockCarla -----------------------------------------------------------------
# ==============================================================================

class MockVehicle:
    """Mock implementation of CARLA vehicle for testing without CARLA"""
    
    def __init__(self):
        self.control = MockVehicleControl()
        self.velocity = MockVector3D(0, 0, 0)
        self.transform = MockTransform()
        self.lights_state = MockVehicleLightState.NONE
    
    def set_light_state(self, lights_state):
        """Mock implementation of set_light_state"""
        # Accept either int or VehicleLightState
        if isinstance(lights_state, int):
            self.lights_state = lights_state
        else:
            self.lights_state = int(lights_state)
        
        # Log individual light states for debugging
        brake_on = bool(self.lights_state & MockVehicleLightState.Brake)
        reverse_on = bool(self.lights_state & MockVehicleLightState.Reverse)
        logging.info(f"[MOCK] Lights - Brake: {brake_on}, Reverse: {reverse_on}")
    
    def get_light_state(self):
        """Mock implementation of get_light_state"""
        return self.lights_state
    
    def apply_control(self, control):
        """Mock implementation of apply_control"""
        self.control = control
        if _tracing: 
            logging.debug(f"[MOCK] Applied control: throttle={control.throttle:.2f}, "
                        f"brake={control.brake:.2f}, steer={control.steer:.2f}, "
                        f"reverse={control.reverse}")
    
    def get_velocity(self):
        """Mock implementation of get_velocity"""
        # Simulate velocity based on control
        if self.control.reverse:
            speed_factor = -1.0
        else:
            speed_factor = 1.0
        
        if self.control.brake > 0.1:
            # Braking - reduce speed
            self.velocity.x = max(0, self.velocity.x - self.control.brake * 0.1)
        else:
            # Accelerating or coasting
            self.velocity.x = min(30.0, self.velocity.x + self.control.throttle * 0.1)
        
        return self.velocity
    
    def get_transform(self):
        """Mock implementation of get_transform"""
        # Update transform based on velocity and steering
        self.transform.location.x += self.velocity.x * 0.05 * math.cos(self.transform.rotation.yaw)
        self.transform.location.y += self.velocity.x * 0.05 * math.sin(self.transform.rotation.yaw)
        self.transform.rotation.yaw += self.control.steer * 0.02
        return self.transform
    
    def get_location(self):
        """Mock implementation of get_location"""
        return self.transform.location
    
    def get_control(self):
        """Mock implementation of get_control"""
        return self.control


class MockVehicleControl:
    """Mock implementation of CARLA VehicleControl"""
    
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.hand_brake = False
        self.reverse = False
        self.manual_gear_shift = False
        self.gear = 1


class MockVector3D:
    """Mock implementation of CARLA Vector3D"""
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockTransform:
    """Mock implementation of CARLA Transform"""
    
    def __init__(self):
        self.location = MockLocation()
        self.rotation = MockRotation()


class MockLocation:
    """Mock implementation of CARLA Location"""
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockRotation:
    """Mock implementation of CARLA Rotation"""
    
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class MockSnapshot:
    """Mock implementation of CARLA WorldSnapshot"""
    
    def __init__(self):
        self.timestamp = MockTimestamp()


class MockTimestamp:
    """Mock implementation of CARLA Timestamp"""
    
    def __init__(self):
        self.elapsed_seconds = 0.0
        self.frame = 0
        self.delta_seconds = FIXED_DELTA_SECONDS
    
    def update(self):
        """Update timestamp for next frame"""
        self.elapsed_seconds += FIXED_DELTA_SECONDS
        self.frame += 1


class MockWorld:
    """Mock implementation of CARLA World for testing"""
    
    def __init__(self):
        self.player = MockVehicle()
        self.snapshot = MockSnapshot()
        self.tick_callbacks = []
    
    def get_snapshot(self):
        """Mock implementation of get_snapshot"""
        return self.snapshot
    
    def tick(self):
        """Mock implementation of tick"""
        self.snapshot.timestamp.update()
        for callback in self.tick_callbacks:
            callback(self.snapshot.timestamp)
        return self.snapshot.timestamp.frame
    
    def on_tick(self, callback):
        """Mock implementation of on_tick"""
        self.tick_callbacks.append(callback)
    
    def get_map(self):
        """Mock implementation of get_map"""
        return MockMap()
    
    def get_actors(self):
        """Mock implementation of get_actors"""
        return MockActorList()


class MockMap:
    """Mock implementation of CARLA Map"""
    
    def __init__(self):
        self.name = "MockTown"
    
    def get_spawn_points(self):
        """Mock implementation of get_spawn_points"""
        return [MockTransform()]


class MockActorList:
    """Mock implementation of CARLA ActorList"""
    
    def __init__(self):
        self.actors = [MockVehicle()]
    
    def filter(self, pattern):
        """Mock implementation of filter"""
        return self.actors


class MockVehicleLightState:
    """Mock implementation of CARLA VehicleLightState"""
    NONE = 0
    Brake = 1
    Reverse = 2
    
    def __new__(cls, value=0):
        """
        Allow instantiation with a value to match CARLA's behavior.
        Returns the integer value directly.
        """
        return int(value)


# Create mock carla module
class MockCarla:
    """Mock implementation of CARLA module"""
    VehicleControl = MockVehicleControl
    VehicleLightState = MockVehicleLightState
    Location = MockLocation
    Rotation = MockRotation
    Transform = MockTransform
    
    class ColorConverter:
        Raw = 0
        Depth = 1
        LogarithmicDepth = 2
        CityScapesPalette = 3


# Replace carla with mock if not available
try:
    import carla
except ImportError:
    logging.warning("CARLA module not found. Using mock implementation.")
    carla = MockCarla()


# ==============================================================================
# -- TestMode ------------------------------------------------------------------
# ==============================================================================

class TestMode:
    """Test mode for Virtual Vehicle without CARLA"""
    
    def __init__(self, args):
        self.args = args
        self.input_manager = None
        self.zenoh_communicator = None
        self.mock_world = None
        self.vehicle_controller = None
        self.status_publisher = None
        self.display = None
        self.clock = None
        self.running = True
    
    def initialize(self):
        """Initialize test components"""
        # Load signal definitions
        if not SignalDefinitions.load_from_file(self.args.signals_config):
            logging.error("Failed to load signal definitions. Exiting.")
            return False
        
        # Initialize pygame for input handling
        pygame.init()
        pygame.font.init()
        
        # Setup minimal display
        self.display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Virtual Vehicle Test Mode")
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Initialize Zenoh
        self.zenoh_communicator = ZenohCommunicator(self.args.router)
        
        # Initialize input manager
        self.input_manager = InputManager()
        
        # Create mock world and vehicle
        self.mock_world = MockWorld()
        
        # Initialize vehicle controller with mock vehicle
        self.vehicle_controller = VehicleController(self.mock_world.player, self.zenoh_communicator)
        
        # Initialize status publisher with mock vehicle
        self.status_publisher = StatusPublisher(self.mock_world.player, self.zenoh_communicator)
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        
        logging.info("Test mode initialized")
        return True
    
    def run(self):
        """Run test loop"""
        try:
            while self.running:               
                # Process input
                input_state = self.input_manager.process_input(self.clock.get_time())
                if input_state.get('quit', False):
                    self.running = False
                    break

                # Tick mock world
                self.mock_world.tick()
                
                # Update vehicle controller
                self.vehicle_controller.update(input_state)
                
                # Publish vehicle status
                self.status_publisher.update(self.mock_world.snapshot.timestamp)
                
                # Render test UI
                self.render_test_ui()
                
                # Cap at 60 FPS
                self.clock.tick(60)
        
        except KeyboardInterrupt:
            logging.info("Cancelled by user")
        
        except Exception as e:
            logging.error(f"Error in test loop: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def render_test_ui(self):
        """Render test UI with input and Zenoh state"""
        self.display.fill((0, 0, 0))
        
        # Render title
        title = self.font.render("Virtual Vehicle Test Mode", True, (255, 255, 255))
        self.display.blit(title, (10, 10))
        
        # Get input states
        input_states = self.vehicle_controller.get_user_input_state()

        # Render user input state
        x_pos_c1 = 10
        y_pos = 50
        self.display.blit(self.font.render(f"User input requests:", True, (255, 255, 255)), (x_pos_c1, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Acc Pedal:  {input_states['acc_pedal']:.2f}", True, (255, 255, 255)), (x_pos_c1, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Brake Pedal: {input_states['brake_pedal']:.2f}", True, (255, 255, 255)), (x_pos_c1, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Steering:    {input_states['steer']:.2f}", True, (255, 255, 255)), (x_pos_c1, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Cruise Control req: {input_states['cc_engage_req']}", True, (255, 255, 255)), (x_pos_c1, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Reverse req: {input_states['reverse_req']}", True, (255, 255, 255)), (x_pos_c1, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Delta Speed: {input_states['delta_speed']:.2f}", True, (255, 255, 255)), (x_pos_c1, y_pos))

        # Render commands applied to vehicle
        x_pos_c2 = 230
        y_pos = 50
        self.display.blit(self.font.render(f"Commands applied to vehicle:", True, (255, 255, 255)), (x_pos_c2, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Throttle: {self.mock_world.player.control.throttle:.2f}", True, (255, 255, 255)), (x_pos_c2, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Brake: {self.mock_world.player.control.brake:.2f}", True, (255, 255, 255)), (x_pos_c2, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Steer: {self.mock_world.player.control.steer:.2f}", True, (255, 255, 255)), (x_pos_c2, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Reverse: {self.mock_world.player.control.reverse}", True, (255, 255, 255)), (x_pos_c2, y_pos))

        brake_on = bool(self.mock_world.player.get_light_state() & carla.VehicleLightState.Brake)
        reverse_on = bool(self.mock_world.player.get_light_state() & carla.VehicleLightState.Reverse)

        y_pos += 25
        self.display.blit(self.font.render(f"Brake lights: {'ON' if brake_on else 'OFF'}", True, (255, 255, 255)), (x_pos_c2, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Reverse lights: {'ON' if reverse_on else 'OFF'}", True, (255, 255, 255)), (x_pos_c2, y_pos))
        
        # Render Zenoh state
        x_pos_c3 = 510
        y_pos = 50
        self.display.blit(self.font.render("Commands received from Zenoh:", True, (255, 255, 255)), (x_pos_c3, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"CC Engaged: {self.zenoh_communicator.vcu_cc_engage_sts}", True, (255, 255, 255)), (x_pos_c3, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Reverse: {self.zenoh_communicator.vcu_reverse_sts}", True, (255, 255, 255)), (x_pos_c3, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Throttle Cmd: {self.zenoh_communicator.vcu_throttle_cmd:.2f}", True, (255, 255, 255)), (x_pos_c3, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"Brake Cmd: {self.zenoh_communicator.vcu_brake_cmd:.2f}", True, (255, 255, 255)), (x_pos_c3, y_pos))
        y_pos += 25
        self.display.blit(self.font.render(f"CC Target Speed: {self.zenoh_communicator.adas_cc_target_speed_sts:.2f}", True, (255, 255, 255)), (x_pos_c3, y_pos))
        
        # Render vehicle state
        y_pos += 80
        velocity = self.mock_world.player.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h
        self.display.blit(self.font.render(f"Speed read from CARLA: {speed:.1f} km/h", True, (255, 255, 255)), (x_pos_c3, y_pos))
        y_pos += 25
        current_speed_kmh = self.status_publisher.get_current_speed_kmh()
        self.display.blit(self.font.render(f"Speed sent to Zenoh: {current_speed_kmh:.1f} km/h", True, (255, 255, 255)), (x_pos_c3, y_pos))
        
        # Render instructions
        y_pos = 300
        self.display.blit(self.font.render("Controls:", True, (255, 200, 0)), (10, y_pos))
        y_pos += 25
        self.display.blit(self.font.render("- WASD/Arrows: Throttle/Brake/Steering", True, (255, 200, 0)), (10, y_pos))
        y_pos += 25
        self.display.blit(self.font.render("- C: Toggle Cruise Control", True, (255, 200, 0)), (10, y_pos))
        y_pos += 25
        self.display.blit(self.font.render("- Q: Toggle Reverse", True, (255, 200, 0)), (10, y_pos))
        y_pos += 25
        self.display.blit(self.font.render("- Z/X: Decrease/Increase Delta Speed", True, (255, 200, 0)), (10, y_pos))
        y_pos += 25
        self.display.blit(self.font.render("- TAB: Change Camera View Angle", True, (255, 200, 0)), (10, y_pos))
        y_pos += 25
        self.display.blit(self.font.render("- ESC: Quit", True, (255, 200, 0)), (10, y_pos))
        
        pygame.display.flip()
    
    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up test resources")
        
        # Close Zenoh
        if self.zenoh_communicator:
            self.zenoh_communicator.close()
        
        # Quit pygame
        pygame.quit()


# ==============================================================================
# -- Utility Functions ---------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_actor_blueprints(world, filter_str, generation):
    bps = world.get_blueprint_library().filter(filter_str)
    if generation.lower() == "all":
        return bps
    if len(bps) == 1:
        return bps
    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("Warning! Invalid generation. No actor spawned.")
            return []
    except:
        print("Warning! Invalid generation. No actor spawned.")
        return []


# ==============================================================================
# -- ZenohCommunicator --------------------------------------------------------
# ==============================================================================

class ZenohCommunicator:
    """Manages Zenoh communication for the Virtual Vehicle"""
    
    def __init__(self, router: str):
        self.session = None
        self.publishers = {}
        self.subscribers = {}
        
        # State from subscribers
        self.vcu_cc_engage_sts = False
        self.vcu_reverse_sts = False
        self.vcu_brake_sts = False
        self.vcu_brake_cmd = 0.0
        self.vcu_throttle_cmd = 0.0
        self.adas_cc_target_speed_sts = 0.0
        self.bcm_reverse_lights_cmd = False
        self.bcm_brake_lights_cmd = False
        
        self._setup_session(router)
        self._setup_publishers()
        self._setup_subscribers()
    
    def _setup_session(self, router: str):
        """Initialize Zenoh session"""
        zenoh_config = zenoh.Config()
        zenoh_config.insert_json5("mode", json.dumps("peer"))
        zenoh_config.insert_json5("connect/endpoints", json.dumps([f"tcp/{router}:7447"]))
        self.session = zenoh.open(zenoh_config)
        logging.info("Zenoh session opened")
    
    def _setup_publishers(self):
        """Create publishers for all provided signals"""
        for signal in SignalDefinitions.get_provided_signals():
            topic = SignalDefinitions.get_topic(signal)
            self.publishers[signal] = self.session.declare_publisher(topic)
            logging.debug(f"Publisher created: {topic}")
    
    def _setup_subscribers(self):
        """Create subscribers for all consumed signals"""
        # VCU Cruise Control Engage Status
        def vcu_cc_engage_callback(sample):
            try:
                payload = sample.payload.to_string().strip().lower()
                self.vcu_cc_engage_sts = payload in ("true", "1", "on")
                logging.debug(f"[Zenoh] vcu_cc_engage_sts: {self.vcu_cc_engage_sts}")
            except Exception as e:
                logging.error(f"vcu_cc_engage_sts callback error: {e}")
        
        # VCU Reverse Status
        def vcu_reverse_sts_callback(sample):
            try:
                payload = sample.payload.to_string().strip().lower()
                self.vcu_reverse_sts = payload in ("true", "1", "on")
                logging.debug(f"[Zenoh] vcu_reverse_sts: {self.vcu_reverse_sts}")
            except Exception as e:
                logging.error(f"vcu_reverse_sts callback error: {e}")
        
        # VCU Brake Status
        def vcu_brake_sts_callback(sample):
            try:
                payload = sample.payload.to_string().strip().lower()
                self.vcu_brake_sts = payload in ("true", "1", "on")
                logging.debug(f"[Zenoh] vcu_brake_sts: {self.vcu_brake_sts}")
            except Exception as e:
                logging.error(f"vcu_brake_sts callback error: {e}")
        
        # VCU Brake Command
        def vcu_brake_cmd_callback(sample):
            try:
                payload = sample.payload.to_string()
                self.vcu_brake_cmd = float(payload)
                if _tracing: logging.debug(f"[Zenoh] vcu_brake_cmd: {self.vcu_brake_cmd}")
            except Exception as e:
                logging.error(f"vcu_brake_cmd callback error: {e}")
        
        # VCU Throttle Command
        def vcu_throttle_cmd_callback(sample):
            try:
                payload = sample.payload.to_string()
                self.vcu_throttle_cmd = float(payload)
                if _tracing: logging.debug(f"[Zenoh] vcu_throttle_cmd: {self.vcu_throttle_cmd}")
            except Exception as e:
                logging.error(f"vcu_throttle_cmd callback error: {e}")
        
        # ADAS Cruise Control Target Speed
        def adas_cc_target_speed_callback(sample):
            try:
                payload = sample.payload.to_string()
                self.adas_cc_target_speed_sts = float(payload)
                logging.debug(f"[Zenoh] adas_cc_target_speed_sts: {self.adas_cc_target_speed_sts}")
            except Exception as e:
                logging.error(f"adas_cc_target_speed_sts callback error: {e}")
        
        # BCM Reverse Lights Command
        def bcm_reverse_lights_cmd_callback(sample):
            try:
                payload = sample.payload.to_string().strip().lower()
                self.bcm_reverse_lights_cmd = payload in ("true", "1", "on")
                logging.debug(f"[Zenoh] bcm_reverse_lights_cmd: {self.bcm_reverse_lights_cmd}")
            except Exception as e:
                logging.error(f"bcm_reverse_lights_cmd callback error: {e}")
        
        # BCM Brake Lights Command
        def bcm_brake_lights_cmd_callback(sample):
            try:
                payload = sample.payload.to_string().strip().lower()
                self.bcm_brake_lights_cmd = payload in ("true", "1", "on")
                logging.debug(f"[Zenoh] bcm_brake_lights_cmd: {self.bcm_brake_lights_cmd}")
            except Exception as e:
                logging.error(f"bcm_brake_lights_cmd callback error: {e}")
        
        # Create all subscribers
        self.subscribers['vcu_cc_engage_sts'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('vcu_cc_engage_sts'), vcu_cc_engage_callback)
        
        self.subscribers['vcu_reverse_sts'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('vcu_reverse_sts'), vcu_reverse_sts_callback)
        
        self.subscribers['vcu_brake_sts'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('vcu_brake_sts'), vcu_brake_sts_callback)
        
        self.subscribers['vcu_brake_cmd'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('vcu_brake_cmd'), vcu_brake_cmd_callback)
        
        self.subscribers['vcu_throttle_cmd'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('vcu_throttle_cmd'), vcu_throttle_cmd_callback)
        
        self.subscribers['adas_cc_target_speed_sts'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('adas_cc_target_speed_sts'), adas_cc_target_speed_callback)
        
        self.subscribers['bcm_reverse_lights_cmd'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('bcm_reverse_lights_cmd'), bcm_reverse_lights_cmd_callback)
        
        self.subscribers['bcm_brake_lights_cmd'] = self.session.declare_subscriber(
            SignalDefinitions.get_topic('bcm_brake_lights_cmd'), bcm_brake_lights_cmd_callback)
    
    def publish(self, signal: str, value: Any):
        """Publish a value to a signal topic"""
        if signal in self.publishers:
            self.publishers[signal].put(str(value))
            
            if  signal == 'ego_reverse_req' or \
                signal == 'ego_cc_engage_req' or \
                signal == 'ego_delta_speed_sts':
                logging.info(f"[Zenoh] Published {signal}: {value}")
            else:
                if _tracing: logging.debug(f"[Zenoh] Published {signal}: {value}")
        else:
            logging.warning(f"Unknown signal: {signal}")
    
    def close(self):
        """Clean up Zenoh resources"""
        for sub in self.subscribers.values():
            sub.undeclare()
        for pub in self.publishers.values():
            pub.undeclare()
        if self.session:
            self.session.close()
        logging.info("Zenoh session closed")


# ==============================================================================
# -- InputManager --------------------------------------------------------------
# ==============================================================================

class InputManager:
    """Manages input from steering wheel and keyboard"""
    
    def __init__(self):
        self.joystick = None
        self.wheel_config = None
        self.steer_cache = 0.0
        
        # Input state
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.handbrake = False
        self.reverse_toggle_requested = False
        self.low_gear_change = False
        self.up_gear_change = False
        self.cc_toggle_requested = False
        self.delta_speed_change = 0.0  # +/- km/h
        self.toggle_camera_view_change = False
        
        self._setup_steering_wheel()
    
    def _setup_steering_wheel(self):
        """Initialize steering wheel if available"""
        pygame.joystick.init()
        
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            
            config_path = Path(__file__).with_name('wheel_config.ini')
            if config_path.exists():
                parser = ConfigParser()
                parser.read(config_path, encoding='utf-8')
                
                self.wheel_config = {
                    'steer_idx': int(parser.get('G29 Racing Wheel', 'steering_wheel')),
                    'throttle_idx': int(parser.get('G29 Racing Wheel', 'throttle')),
                    'brake_idx': int(parser.get('G29 Racing Wheel', 'brake')),
                    'reverse_idx': int(parser.get('G29 Racing Wheel', 'reverse')),
                    'handbrake_idx': int(parser.get('G29 Racing Wheel', 'handbrake')),
                    'up_gear_idx': int(parser.get('G29 Racing Wheel', 'up_gear')),
                    'low_gear_idx': int(parser.get('G29 Racing Wheel', 'low_gear')),
                    'cruise_toggle_idx': int(parser.get('G29 Racing Wheel', 'toggle_cruise_control')),
                    'cruise_inc_idx': int(parser.get('G29 Racing Wheel', 'cruise_control_increase')),
                    'cruise_dec_idx': int(parser.get('G29 Racing Wheel', 'cruise_control_decrease')),
                    'camera_idx': int(parser.get('G29 Racing Wheel', 'toggle_camera'))
                }
                logging.info("Steering wheel configured")
            else:
                logging.warning("wheel_config.ini not found")
    
    def process_input(self, milliseconds: int) -> Dict[str, Any]:
        """Process input from steering wheel and keyboard"""
        # Reset toggle flags
        self.reverse_toggle_requested = False
        self.low_gear_change = False
        self.up_gear_change = False
        self.cc_toggle_requested = False
        self.delta_speed_change = 0.0
        self.toggle_camera_view_change = False
        
        quit_requested = False
        
        # Process keyboard and wheel button events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            
            # Steering wheel button events
            elif event.type == pygame.JOYBUTTONDOWN and self.wheel_config:
                if event.button == self.wheel_config['cruise_dec_idx']:
                    self.delta_speed_change = -5.0
                    logging.debug("'cruise_dec_idx' button pressed")
                elif event.button == self.wheel_config['cruise_inc_idx']:
                    self.delta_speed_change = 5.0
                    logging.debug("'cruise_inc_idx' button pressed")
                elif event.button == self.wheel_config['camera_idx']:
                    self.toggle_camera_view_change = True
                    logging.debug("'camera_idx' button pressed")
                elif event.button == self.wheel_config['cruise_toggle_idx']:
                    self.cc_toggle_requested = True
                    logging.debug("'cruise_toggle_idx' button pressed")
                elif event.button == self.wheel_config['low_gear_idx']:
                    self.low_gear_change = True
                    logging.debug("'low_gear_idx' button pressed")
                elif event.button == self.wheel_config['up_gear_idx']:
                    self.up_gear_change = True
                    logging.debug("'up_gear_idx' button pressed")
            
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    quit_requested = True
                    logging.debug("Quit requested via Esc key")
                elif event.key == K_TAB:
                    self.toggle_camera_view_change = True
                    logging.debug("TAB key pressed")
                elif event.key == K_c and (pygame.key.get_mods() & KMOD_CTRL):
                    quit_requested = True
                    logging.debug("Quit requested via Ctrl+C")
                elif event.key == K_c:
                    self.cc_toggle_requested = True
                    logging.debug("C key pressed")
                elif event.key == K_q:
                    self.reverse_toggle_requested = True
                    logging.debug("Q key pressed")
                elif event.key == K_z:
                    self.delta_speed_change = -5.0
                    logging.debug("Z key pressed")
                elif event.key == K_x:
                    self.delta_speed_change = 5.0
                    logging.debug("X key pressed")
        
        # Check quit after processing all events
        if quit_requested:
            return {
                'quit': True,
                'throttle': 0.0,
                'brake': 0.0,
                'steer': 0.0,
                'handbrake': False,
                'reverse_toggle': False,
                'low_gear': False,
                'up_gear': False,
                'cc_toggle': False,
                'delta_speed': 0.0,
                'toggle_camera_view': False
            }
        
        # Process continuous inputs (steering, throttle, brake)
        keys = pygame.key.get_pressed()
        
        # Keyboard input
        if not self.joystick:
            self.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
            self.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
            
            if self.throttle > 0.0:
                logging.debug("UP or W key pressed")
            elif self.brake > 0.0:
                logging.debug("DOWN or S key pressed")

            steer_increment = 5e-4 * milliseconds
            if keys[K_LEFT] or keys[K_a]:
                self.steer_cache -= steer_increment
                logging.debug("LEFT or A key pressed")
            elif keys[K_RIGHT] or keys[K_d]:
                self.steer_cache += steer_increment
                logging.debug("RIGHT or D key pressed")
            else:
                self.steer_cache = 0.0
            
            self.steer_cache = np.clip(self.steer_cache, -0.7, 0.7)
            self.steer = round(self.steer_cache, 1)
            self.handbrake = keys[K_SPACE]

        # Steering wheel input
        else:
            num_axes = self.joystick.get_numaxes()
            js_inputs = [self.joystick.get_axis(i) for i in range(num_axes)]
            js_buttons = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]
            
            cfg = self.wheel_config
            
            # Steering
            K1 = 0.50
            self.steer = K1 * math.tan(1.1 * js_inputs[cfg['steer_idx']])
            
            # Throttle
            K2 = 1.6
            self.throttle = K2 + (2.05 * math.log10(-0.7 * js_inputs[cfg['throttle_idx']] + 1.4) - 1.2) / 0.92
            self.throttle = np.clip(self.throttle, 0.0, 1.0)
            
            # Brake
            self.brake = 1.6 + (2.05 * math.log10(-0.7 * js_inputs[cfg['brake_idx']] + 1.4) - 1.2) / 0.92
            self.brake = np.clip(self.brake, 0.0, 1.0)
            
            self.handbrake = bool(js_buttons[cfg['handbrake_idx']])
        
        return {
            'quit': False,
            'throttle': self.throttle,
            'brake': self.brake,
            'steer': self.steer,
            'handbrake': self.handbrake,
            'reverse_toggle': self.reverse_toggle_requested,
            'low_gear': self.low_gear_change,
            'up_gear': self.up_gear_change,
            'cc_toggle': self.cc_toggle_requested,
            'delta_speed': self.delta_speed_change,
            'toggle_camera_view': self.toggle_camera_view_change
        }
    
    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- VehicleController ---------------------------------------------------------
# ==============================================================================

class VehicleController:
    """Controls the CARLA vehicle based on inputs and Zenoh commands"""
    
    def __init__(self, vehicle, zenoh_communicator: ZenohCommunicator):
        self._vehicle = vehicle
        self._zenoh = zenoh_communicator
        
        # Check if we're using a mock vehicle
        if hasattr(vehicle, 'control'):
            self._control = vehicle.control
        else:
            self._control = carla.VehicleControl()
        
        self._lights = carla.VehicleLightState.NONE
        
        # Internal state for tracking toggle requests
        self._cc_request_state = False
        self._reverse_request_state = False
        self._delta_speed = 0.0

        self._prev_vcu_cc_engage_sts = False

        # Initialize notification callback
        self._notification_callback = None
    
    def set_notification_callback(self, callback):
        """Set callback for notifications"""
        self._notification_callback = callback
    
    def update(self, input_state: Dict[str, Any]):
        """Update vehicle control based on input and Zenoh commands"""
        # Process cruise control toggle request
        if input_state['cc_toggle']:
            self._cc_request_state = not self._cc_request_state
            self._zenoh.publish('ego_cc_engage_req', 1 if self._cc_request_state else 0)

            if self._notification_callback:
                status = "ON" if self._cc_request_state else "OFF"
                self._notification_callback(f"Cruise Control Request: {status}")

        # If VCU has disengaged cruise control, update EGO's request state
        if self._prev_vcu_cc_engage_sts and not self._zenoh.vcu_cc_engage_sts and self._cc_request_state:    
            self._cc_request_state = False
            self._zenoh.publish('ego_cc_engage_req', 0)

            if self._notification_callback:
                self._notification_callback("Cruise Control Request: OFF (brake pressed)")

        self._prev_vcu_cc_engage_sts = self._zenoh.vcu_cc_engage_sts

        if  (input_state['reverse_toggle']) or \
            (input_state['low_gear'] and self._control.gear >= 0) or \
            (input_state['up_gear'] and self._control.gear <= 0):
            self._reverse_request_state = not self.get_reverse_status()
            self._zenoh.publish('ego_reverse_req', 1 if self._reverse_request_state else 0)
            
            if self._notification_callback:
                status = "ON" if self._reverse_request_state else "OFF"
                self._notification_callback(f"Reverse Mode Request: {status}")

        # Process delta speed changes        
        if input_state['delta_speed'] != 0:
            self._delta_speed = input_state['delta_speed']
            self._zenoh.publish('ego_delta_speed_sts', self._delta_speed)
            
            if self._notification_callback:
                # self._delta_speed = input_state['delta_speed']
                self._notification_callback(f"Speed Adjustment: {self._delta_speed:+.1f} km/h")
        
        # Update user input commands
        self._ego_acc_pedal_sts = input_state['throttle']
        self._ego_brake_pedal_sts = input_state['brake']
        self._ego_steering_sts = input_state['steer']
       
        # Publish manual inputs to Zenoh
        self._zenoh.publish('ego_acc_pedal_sts', input_state['throttle'])
        self._zenoh.publish('ego_brake_pedal_sts', input_state['brake'])

        # Apply control based on vcu commands for throttle and brake
        self._control.throttle = self._zenoh.vcu_throttle_cmd
        self._control.brake = self._zenoh.vcu_brake_cmd        

        # Always use manual steering
        self._control.steer = input_state['steer']
        self._control.hand_brake = input_state['handbrake']
        
        # Set reverse based on VCU status
        self._control.reverse = self._zenoh.vcu_reverse_sts
        self._control.gear = -1 if self._zenoh.vcu_reverse_sts else 1
        
        # Update lights based on BCM commands
        current_lights = self._lights
        if self._zenoh.bcm_brake_lights_cmd:
            current_lights |= carla.VehicleLightState.Brake
        else:
            current_lights &= ~carla.VehicleLightState.Brake
        
        if self._zenoh.bcm_reverse_lights_cmd:
            current_lights |= carla.VehicleLightState.Reverse
        else:
            current_lights &= ~carla.VehicleLightState.Reverse
        
        if current_lights != self._lights:
            self._lights = current_lights
            try:
                self._vehicle.set_light_state(carla.VehicleLightState(self._lights))
                logging.info(f"self._lights: {self._lights}")
                logging.info(f"bcm_brake_lights_cmd: {self._zenoh.bcm_brake_lights_cmd}")
                logging.info(f"bcm_reverse_lights_cmd: {self._zenoh.bcm_reverse_lights_cmd}")
            except Exception as e:
                logging.error(f"Could not set light state: {e}")
        
        # Apply control to vehicle
        try:
            self._vehicle.apply_control(self._control)
        except Exception as e:
            logging.debug(f"Could not apply control: {e}")
        
        if _tracing: 
            logging.debug(f"[Control] throttle={self._control.throttle:.2f}, "
                        f"steer={self._control.steer:.2f}, brake={self._control.brake:.2f}, "
                        f"reverse={self._control.reverse}")
    
    def get_cruise_control_status(self) -> bool:
        """Get cruise control status from VCU"""
        return self._zenoh.vcu_cc_engage_sts
    
    def get_reverse_status(self) -> bool:
        """Get reverse status from VCU"""
        return self._zenoh.vcu_reverse_sts

    def get_user_input_state(self) -> Dict[str, Any]:
        """Get current user input state"""
        return {
            'acc_pedal': self._ego_acc_pedal_sts,
            'brake_pedal': self._ego_brake_pedal_sts,
            'steer': self._ego_steering_sts,
            'cc_engage_req': self._cc_request_state,
            'reverse_req': self._reverse_request_state,
            'delta_speed': self._delta_speed
        }

# ==============================================================================
# -- StatusPublisher -----------------------------------------------------------
# ==============================================================================

class StatusPublisher:
    """Publishes vehicle status to Zenoh"""
    
    def __init__(self, vehicle, zenoh_communicator: ZenohCommunicator):
        self._vehicle = vehicle
        self._zenoh = zenoh_communicator
        self._current_speed_kmh = 0.0
    
    def update(self, timestamp):
        """Publish vehicle status (absolute clock and velocity) to Zenoh"""
        try:
            # Publish absolute elapsed time in seconds
            # This is the simulation clock that increases monotonically
            elapsed_seconds = timestamp.elapsed_seconds
            self._zenoh.publish('sim_clock_sts', elapsed_seconds)
            
            # Publish velocity
            velocity = self._vehicle.get_velocity()
            self._current_speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            self._zenoh.publish('ego_velocity_sts', self._current_speed_kmh)
            
            if _tracing: logging.debug(f"[Status] clock={elapsed_seconds:.3f}s, velocity={self._current_speed_kmh:.1f} km/h")
        except Exception as e:
            logging.error(f"Error publishing status: {e}")

    def get_current_speed_kmh(self):
        """Get the current speed in km/h"""
        return self._current_speed_kmh
        
# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World:
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._zenoh_mgr = None
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._fullscreen = args.fullscreen
        self.restart()
        self.world.on_tick(hud.on_world_tick)
    

    def set_zenoh_manager(self, zenoh_mgr: ZenohCommunicator):
        """Attach Zenoh manager and create radar sensor for current ego vehicle."""
        self._zenoh_mgr = zenoh_mgr
        if self.player is not None:
            self._create_radar_sensor()

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("No blueprints found with specified filters")
        
        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '255, 255, 255')
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        
        # Spawn player
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        
        # Set up sensors
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self._create_radar_sensor()
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
    

    def _create_radar_sensor(self):
        """Create or recreate the radar sensor for the current player."""
        if self.radar_sensor is not None:
            self.radar_sensor.destroy()
            self.radar_sensor = None

        if self._zenoh_mgr is not None and self.player is not None:
            self.radar_sensor = RadarDistanceSensor(self.player, self._zenoh_mgr, self.hud)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification(f'Weather: {preset[1]}')
        self.player.get_world().set_weather(preset[0])
    
    def tick(self, clock):
        self.hud.tick(self, clock)
    
    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)
    
    def destroy(self):
        sensors = [
            self.camera_manager.sensor if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor if self.gnss_sensor else None,
            self.radar_sensor.sensor if self.radar_sensor else None
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()

        self.camera_manager = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.radar_sensor = None

        if self.player is not None:
            self.player.destroy()
            self.player = None

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD:
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._vehicle_controller = None
    
    def set_vehicle_controller(self, controller):
        self._vehicle_controller = controller
    
    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        
        vehicles = world.world.get_actors().filter('vehicle.*')
        
        self._info_text = [
            f'Server:  {self.server_fps:16.0f} FPS',
            f'Client:  {clock.get_fps():16.0f} FPS',
            '',
            f'Vehicle: {get_actor_display_name(world.player, truncate=20):20s}',
            f'Map:     {world.world.get_map().name.split("/")[-1]:20s}',
            f'Simulation time: {datetime.timedelta(seconds=int(self.simulation_time))}',
            '',
            f'Speed:   {3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2):15.0f} km/h',
            f'Heading:{t.rotation.yaw:16.0f}\N{DEGREE SIGN} {heading:2s}',
            f'Location:{t.location.x:5.1f}, {t.location.y:5.1f}',
            f'GNSS:{world.gnss_sensor.lat:2.6f}, {world.gnss_sensor.lon:3.6f}',
            f'Height:  {t.location.z:18.0f} m',
            ''
        ]
        
        if isinstance(c, carla.VehicleControl):
            cc_status = "ON" if (self._vehicle_controller and 
                               self._vehicle_controller.get_cruise_control_status()) else "OFF"
            reverse_status = "ON" if (self._vehicle_controller and 
                                    self._vehicle_controller.get_reverse_status()) else "OFF"
            radar_value = world.radar_sensor.last_discrete_value if world.radar_sensor else 0
            radar_distance = world.radar_sensor.last_distance_m if world.radar_sensor else -1.0
            
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                f'Reverse:      {reverse_status}',
                f'Hand brake:   {"ON" if c.hand_brake else "OFF"}',
                f'Manual:       {"ON" if c.manual_gear_shift else "OFF"}',
                f'Cruise Ctrl:  {cc_status}',
                f'Front Radar:  {radar_value}',
                f'Radar Dist:   {radar_distance:6.2f} m',
                f'Gear:         {"D" if c.gear > 0 else {-1: "R", 0: "N"}.get(c.gear, c.gear)}'
            ]
        
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            f'Number of vehicles: {len(vehicles):8d}'
        ]
        
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt(
                (l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2
            )
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append(f'{d:4.0f}m {vehicle_type}')
    
    def toggle_info(self):
        self._show_info = not self._show_info
    
    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)
    
    def error(self, text):
        self._notifications.set_text(f'Error: {text}', (255, 0, 0))
    
    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText:
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
    
    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))
    
    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)
    
    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText:
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
        self._render = False
        self.surface.set_alpha(220)
    
    def toggle(self):
        self._render = not self._render
    
    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- Sensors -------------------------------------------------------------------
# ==============================================================================

class CollisionSensor:
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
    
    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    
    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f'Collision with {actor_type!r}')
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor:
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
    
    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = [f'{str(x).split()[-1]!r}' for x in lane_types]
        self.hud.notification(f'Crossed line {" and ".join(text)}')


class GnssSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))
    
    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude



class RadarDistanceSensor:
    """Radar sensor that publishes discretized front distance via Zenoh."""

    def __init__(self, parent_actor, zenoh_communicator: ZenohCommunicator, hud=None):
        self.sensor = None
        self._parent = parent_actor
        self._zenoh = zenoh_communicator
        self._hud = hud
        self.last_discrete_value = 0
        self.last_distance_m = -1.0

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('range', str(RADAR_RANGE_METERS))
        bp.set_attribute('horizontal_fov', str(RADAR_HORIZONTAL_FOV_DEG))
        bp.set_attribute('vertical_fov', str(RADAR_VERTICAL_FOV_DEG))
        bp.set_attribute('points_per_second', str(RADAR_POINTS_PER_SECOND))
        bp.set_attribute('sensor_tick', str(RADAR_SENSOR_TICK))

        radar_transform = carla.Transform(
            carla.Location(x=2.2, z=1.0),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        self.sensor = world.spawn_actor(bp, radar_transform, attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: RadarDistanceSensor._on_radar_event(weak_self, data))

    @staticmethod
    def _discretize_distance(distance_m: float) -> int:
        if distance_m < 0:
            return 0
        if distance_m < 5.0:
            return 3
        if distance_m < 10.0:
            return 2
        if distance_m < 15.0:
            return 1
        return 0

    @staticmethod
    def _on_radar_event(weak_self, radar_data):
        self = weak_self()
        if not self:
            return

        min_depth = None
        max_az = math.radians(RADAR_CENTER_AZIMUTH_DEG)
        max_alt = math.radians(RADAR_CENTER_ALTITUDE_DEG)

        for detection in radar_data:
            if abs(detection.azimuth) > max_az:
                continue
            if abs(detection.altitude) > max_alt:
                continue

            if min_depth is None or detection.depth < min_depth:
                min_depth = detection.depth

        if min_depth is None:
            self.last_distance_m = -1.0
            discrete_value = 0
        else:
            self.last_distance_m = float(min_depth)
            discrete_value = self._discretize_distance(self.last_distance_m)

        self.last_discrete_value = discrete_value
        self._zenoh.publish('front_distance_discrete', discrete_value)

        if self.last_distance_m <= 2:
            self._zenoh.publish('aeb_alert', 1)
        else:
            self._zenoh.publish('aeb_alert', 0)

        logging.debug(
            f"[Radar] front_distance={self.last_distance_m:.3f} m, "
            f"discrete={self.last_discrete_value}"
        )

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
class CameraManager:
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=0.3, y=0.0, z=1.3), carla.Rotation(pitch=0.0))
        ]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']
        ]
        
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None
    
    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])
    
    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            self.sensors[index][0] != self.sensors[self.index][0]
        
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index
    
    def next_sensor(self):
        self.set_sensor(self.index + 1)
    
    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification(f'Recording {"On" if self.recording else "Off"}')
    
    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- VirtualVehicle ------------------------------------------------------------
# ==============================================================================

class VirtualVehicle:
    """Main class that orchestrates the Virtual Vehicle Module"""
    
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.input_manager = None
        self.zenoh_communicator = None
        self.vehicle_controller = None
        self.status_publisher = None
        self.hud = None
        self.display = None
        self.clock = None
        self.original_settings = None
    
        self.sim_world = None
        self.traffic_manager = None
        self.spawned_traffic_vehicles = []

    def initialize(self):
        """Initialize all components"""
        # Load signal definitions
        if not SignalDefinitions.load_from_file(self.args.signals_config):
            logging.error("Failed to load signal definitions. Exiting.")
            return False
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Setup display
        if self.args.fullscreen:
            self.display = pygame.display.set_mode(
                (0, 0),
                pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            info = pygame.display.Info()
            self.args.width, self.args.height = info.current_w, info.current_h
        else:
            self.display = pygame.display.set_mode(
                (self.args.width, self.args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
        
        self.display.fill((0, 0, 0))
        pygame.display.flip()
        
        # Connect to CARLA
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(2000.0)
        
        self.sim_world = self.client.get_world()
        
        blueprint_library = self.sim_world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter("vehicle.*")
        spawn_points = self.sim_world.get_map().get_spawn_points()
        
        if len(spawn_points) < TRAFFIC_NUM_VEHICLES:
            raise RuntimeError(
                f"There are only {len(spawn_points)} spawn points in the map, "
                f"but {TRAFFIC_NUM_VEHICLES} traffic vehicles were requested."
            )
        
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.global_percentage_speed_difference(0.0)
        
        self.original_settings = self.sim_world.get_settings()
        
        random.shuffle(spawn_points)
        
        for i in range(TRAFFIC_NUM_VEHICLES):
            bp = random.choice(vehicle_blueprints)
            
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            
            transform = spawn_points[i]
            vehicle = self.sim_world.try_spawn_actor(bp, transform)
            
            if vehicle is not None:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.traffic_manager.update_vehicle_lights(vehicle, True)
                self.spawned_traffic_vehicles.append(vehicle)
        
        logging.info(f"{len(self.spawned_traffic_vehicles)} traffic vehicles spawned with Traffic Manager.")
        
        # Configure synchronous mode
        if self.args.sync:
            settings = self.sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
            self.sim_world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
        
        # Initialize Zenoh
        self.zenoh_communicator = ZenohCommunicator(self.args.router)
        
        # Initialize HUD
        self.hud = HUD(self.args.width, self.args.height)
        
        # Initialize world
        self.world = World(self.sim_world, self.hud, self.args)
        self.world.set_zenoh_manager(self.zenoh_communicator)
        
        # Initialize input manager
        self.input_manager = InputManager()
        
        # Initialize vehicle controller
        self.vehicle_controller = VehicleController(self.world.player, self.zenoh_communicator)
        self.vehicle_controller.set_notification_callback(self.hud.notification)
        
        # Initialize status publisher
        self.status_publisher = StatusPublisher(self.world.player, self.zenoh_communicator)
        
        # Set vehicle controller for HUD
        self.hud.set_vehicle_controller(self.vehicle_controller)
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        
        # Initial tick
        if self.args.sync:
            self.sim_world.tick()
        else:
            self.sim_world.wait_for_tick()
        
        self.hud.notification("Virtual Vehicle Module initialized", seconds=3.0)
        return True
    
    def run(self):
        """Main loop"""
        try:
            while True:
                # Tick the simulation
                if self.args.sync:
                    self.world.world.tick()
                
                self.clock.tick_busy_loop(60)
                
                # Process input
                input_state = self.input_manager.process_input(self.clock.get_time())
                if input_state.get('quit', False):
                    break
                
                # Handle camera controls
                if input_state.get('toggle_camera_view', False):
                    self.world.camera_manager.toggle_camera()
                
                if input_state.get('next_camera_sensor', False):
                    self.world.camera_manager.next_sensor()
                
                if input_state.get('toggle_help', False):
                    self.world.hud.help.toggle()
                
                # Get current timestamp
                snapshot = self.world.world.get_snapshot()
                timestamp = snapshot.timestamp
                
                # Update vehicle controller
                self.vehicle_controller.update(input_state)
                
                # Publish vehicle status
                self.status_publisher.update(timestamp)
                
                # Update and render
                self.world.tick(self.clock)
                self.world.render(self.display)
                pygame.display.flip()
        
        except KeyboardInterrupt:
            logging.info("Cancelled by user")
        
        except Exception as e:
            logging.error(f"Error in game loop: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up resources")
        
        # Restore original settings
        if self.original_settings and self.sim_world:
            self.sim_world.apply_settings(self.original_settings)
        
        # Destroy world
        if self.world:
            self.world.destroy()
        
        for vehicle in self.spawned_traffic_vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except Exception:
                pass
        
        # Close Zenoh
        if self.zenoh_communicator:
            self.zenoh_communicator.close()
        
        # Quit pygame
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """Entry point"""
    argparser = argparse.ArgumentParser(
        description='X-Verse Virtual Vehicle Module'
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information'
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='192.168.1.102',
        help='IP of the CARLA host server (default: 192.168.1.102)'
    )
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)'
    )
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")'
    )
    argparser.add_argument(
        '--router',
        default='192.168.1.102',
        type=str,
        help='IP address of the Zenoh router (default: 192.168.1.102)'
    )
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='Restrict to certain actor generation (values: "1","2","All" - default: "2")'
    )
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='ego_vehicle',
        help='Actor role name (default: "ego_vehicle")'
    )
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution'
    )
    argparser.add_argument(
        '--fullscreen',
        default=False,
        type=bool,
        help='Activate fullscreen mode'
    )
    argparser.add_argument(
        '--signals-config',
        default='signals_config.json',
        help='Path to signal configuration file (default: signals_config.json)'
    )
    argparser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode without CARLA'
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    # Setup logging
    log_level = logging.DEBUG if args.debug or args.test_mode else logging.INFO
    logging.basicConfig(
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
        level=log_level,
        force=True  # Force reconfiguration if already configured
    )
    
    # Set level for root logger explicitly
    logging.getLogger().setLevel(log_level)
    
    # Setup CARLA import based on test mode
    global carla, cc
    carla_module, cc_module = setup_carla_import(args.test_mode)
    
    if carla_module is None:
        # Use mock implementation
        carla = MockCarla()
        cc = MockCarla.ColorConverter
    else:
        carla = carla_module
        cc = cc_module

    # Log startup information
    logging.info("=" * 60)
    logging.info("X-Verse Virtual Vehicle Module")
    logging.info("=" * 60)
    logging.info(f'Zenoh router: {args.router}')
    logging.info(f'Signal configuration: {args.signals_config}')
    logging.info(f'Debug mode: {args.debug}')
    
    # Print help text
    print(__doc__)
    
    try:
        if args.test_mode:
            logging.info("Running in test mode (without CARLA)")
            test_mode = TestMode(args)
            if test_mode.initialize():
                test_mode.run()
        else:
            # Validate CARLA is available for normal mode
            if carla_module is None:
                logging.error("CARLA is required for normal mode but could not be imported")
                logging.error("Please install CARLA or use --test-mode flag")
                sys.exit(1)
            
            logging.info(f'Connecting to CARLA server {args.host}:{args.port}')
            virtual_vehicle = VirtualVehicle(args)
            if virtual_vehicle.initialize():
                virtual_vehicle.run()
    except KeyboardInterrupt:
        logging.info('\nCancelled by user. Bye!')
    except Exception as e:
        logging.error(f'Fatal error: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
