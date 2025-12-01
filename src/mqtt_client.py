"""
MQTT Client - Home Assistant MQTT Discovery integration.

This module provides MQTT publishing for Home Assistant integration:
1. Auto-discovery of entities via MQTT Discovery protocol
2. Real-time state publishing
3. Availability management with Last Will and Testament
"""

import json
import logging
import ssl
import threading
import time
from typing import Optional, Callable

from .config import MqttConfig
from .state_snapshot import StateSnapshot

# Try to import paho-mqtt
try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    mqtt = None

logger = logging.getLogger(__name__)


class MqttPublisher:
    """
    MQTT client for publishing baby monitor state to Home Assistant.

    Implements Home Assistant MQTT Discovery for automatic entity creation.
    """

    def __init__(self, config: MqttConfig):
        """
        Initialize MQTT publisher.

        Args:
            config: MQTT configuration

        Raises:
            ImportError: If paho-mqtt is not installed
        """
        if not PAHO_AVAILABLE:
            raise ImportError(
                "paho-mqtt is required for MQTT support. "
                "Install with: pip install paho-mqtt"
            )

        self._config = config
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._lock = threading.Lock()

        # Topics
        self._availability_topic = f"{config.base_topic}/availability"

        # Device info for Home Assistant
        self._device_info = {
            "identifiers": [config.device_id],
            "name": config.device_name,
            "manufacturer": "baby_awake_detection",
            "model": "crib_pose_monitor",
            "sw_version": "1.0.0",
        }

    def connect(self) -> bool:
        """
        Connect to the MQTT broker.

        Sets up Last Will and Testament for offline detection.

        Returns:
            True if connection initiated successfully
        """
        try:
            # Create client with callback API version
            self._client = mqtt.Client(
                client_id=self._config.client_id,
                clean_session=True,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )

            # Set callbacks
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            # Set authentication if provided
            if self._config.username:
                self._client.username_pw_set(
                    self._config.username,
                    self._config.password,
                )

            # Set TLS if enabled
            if self._config.tls:
                self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

            # Set Last Will and Testament (LWT)
            self._client.will_set(
                self._availability_topic,
                payload="offline",
                qos=1,
                retain=True,
            )

            # Connect (non-blocking)
            logger.info(f"Connecting to MQTT broker at {self._config.host}:{self._config.port}")
            self._client.connect_async(
                self._config.host,
                self._config.port,
                keepalive=60,
            )

            # Start network loop in background thread
            self._client.loop_start()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """Callback when connected to broker."""
        if reason_code == 0:
            logger.info("Connected to MQTT broker")
            with self._lock:
                self._connected = True

            # Publish availability and discovery
            self.publish_availability(online=True)
            self.publish_discovery()
        else:
            logger.error(f"MQTT connection failed with code: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        """Callback when disconnected from broker."""
        logger.warning(f"Disconnected from MQTT broker (code: {reason_code})")
        with self._lock:
            self._connected = False

    def _on_message(self, client, userdata, message):
        """Callback for received messages (not used but required)."""
        pass

    def publish_availability(self, online: bool):
        """
        Publish availability status.

        Args:
            online: True for online, False for offline
        """
        if not self._client:
            return

        payload = "online" if online else "offline"
        try:
            self._client.publish(
                self._availability_topic,
                payload=payload,
                qos=1,
                retain=True,
            )
            logger.debug(f"Published availability: {payload}")
        except Exception as e:
            logger.error(f"Failed to publish availability: {e}")

    def publish_discovery(self):
        """
        Publish Home Assistant MQTT Discovery messages for all entities.

        Creates retained config messages that Home Assistant uses to
        automatically create entities.
        """
        if not self._client:
            return

        logger.info("Publishing Home Assistant discovery messages")

        # Define all entities
        entities = self._get_discovery_entities()

        for entity in entities:
            topic = entity["topic"]
            payload = entity["payload"]

            try:
                self._client.publish(
                    topic,
                    payload=json.dumps(payload),
                    qos=1,
                    retain=True,
                )
                logger.debug(f"Published discovery: {topic}")
            except Exception as e:
                logger.error(f"Failed to publish discovery for {topic}: {e}")

    def _get_discovery_entities(self) -> list:
        """
        Get all entity discovery configurations.

        Returns:
            List of dicts with 'topic' and 'payload' keys
        """
        prefix = self._config.discovery_prefix
        device_id = self._config.device_id
        base = self._config.base_topic

        entities = []

        # 1. Binary sensor - Baby Present (occupancy)
        entities.append({
            "topic": f"{prefix}/binary_sensor/{device_id}_present/config",
            "payload": {
                "name": "Baby Present",
                "unique_id": f"{device_id}_present",
                "state_topic": f"{base}/present",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "occupancy",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 2. Binary sensor - Baby Awake
        entities.append({
            "topic": f"{prefix}/binary_sensor/{device_id}_awake/config",
            "payload": {
                "name": "Baby Awake",
                "unique_id": f"{device_id}_awake",
                "state_topic": f"{base}/awake",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "occupancy",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 3. Sensor - Current State (enum)
        entities.append({
            "topic": f"{prefix}/sensor/{device_id}_state/config",
            "payload": {
                "name": "Baby State",
                "unique_id": f"{device_id}_state",
                "state_topic": f"{base}/state",
                "device_class": "enum",
                "options": ["empty", "present", "asleep", "awake"],
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 4. Sensor - Movement Combined
        entities.append({
            "topic": f"{prefix}/sensor/{device_id}_movement/config",
            "payload": {
                "name": "Baby Movement",
                "unique_id": f"{device_id}_movement",
                "state_topic": f"{base}/movement/combined",
                "unit_of_measurement": "",
                "state_class": "measurement",
                "icon": "mdi:motion-sensor",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 5. Sensor - Movement Torso
        entities.append({
            "topic": f"{prefix}/sensor/{device_id}_movement_torso/config",
            "payload": {
                "name": "Torso Movement",
                "unique_id": f"{device_id}_movement_torso",
                "state_topic": f"{base}/movement/torso",
                "unit_of_measurement": "",
                "state_class": "measurement",
                "icon": "mdi:human",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 6. Sensor - Movement Arms
        entities.append({
            "topic": f"{prefix}/sensor/{device_id}_movement_arms/config",
            "payload": {
                "name": "Arm Movement",
                "unique_id": f"{device_id}_movement_arms",
                "state_topic": f"{base}/movement/arms",
                "unit_of_measurement": "",
                "state_class": "measurement",
                "icon": "mdi:arm-flex",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 7. Sensor - Movement Smoothed
        entities.append({
            "topic": f"{prefix}/sensor/{device_id}_movement_smoothed/config",
            "payload": {
                "name": "Smoothed Movement",
                "unique_id": f"{device_id}_movement_smoothed",
                "state_topic": f"{base}/movement/smoothed",
                "unit_of_measurement": "",
                "state_class": "measurement",
                "icon": "mdi:motion-sensor",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 8. Sensor - Pose Quality
        entities.append({
            "topic": f"{prefix}/sensor/{device_id}_pose_quality/config",
            "payload": {
                "name": "Pose Quality",
                "unique_id": f"{device_id}_pose_quality",
                "state_topic": f"{base}/pose/quality",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "icon": "mdi:quality-high",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        # 9. Sensor - Pose Class (enum)
        entities.append({
            "topic": f"{prefix}/sensor/{device_id}_pose_class/config",
            "payload": {
                "name": "Pose Class",
                "unique_id": f"{device_id}_pose_class",
                "state_topic": f"{base}/pose/class",
                "device_class": "enum",
                "options": [
                    "unknown",
                    "lying_flat",
                    "curled_tucked",
                    "side_lying",
                    "on_elbows",
                    "on_all_fours",
                    "sitting",
                    "standing",
                ],
                "icon": "mdi:human-child",
                "availability_topic": self._availability_topic,
                "device": self._device_info,
            },
        })

        return entities

    def publish_state(self, snapshot: StateSnapshot):
        """
        Publish current state to all MQTT topics.

        Args:
            snapshot: Current state snapshot
        """
        if not self._client or not self._connected:
            return

        base = self._config.base_topic
        qos = self._config.qos

        # Build all messages
        messages = [
            (f"{base}/present", "ON" if snapshot.present else "OFF"),
            (f"{base}/awake", "ON" if snapshot.awake else "OFF"),
            (f"{base}/state", snapshot.state_name),
            (f"{base}/movement/torso", f"{snapshot.movement_torso:.3f}"),
            (f"{base}/movement/arms", f"{snapshot.movement_arms:.3f}"),
            (f"{base}/movement/combined", f"{snapshot.movement_combined:.3f}"),
            (f"{base}/movement/smoothed", f"{snapshot.movement_smoothed:.3f}"),
            (f"{base}/pose/quality", f"{snapshot.pose_quality:.1f}"),
            (f"{base}/pose/class", snapshot.pose_class_name),
        ]

        # Publish all messages (no retain for state topics)
        for topic, payload in messages:
            try:
                self._client.publish(topic, payload=payload, qos=qos, retain=False)
            except Exception as e:
                logger.error(f"Failed to publish to {topic}: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to broker."""
        with self._lock:
            return self._connected

    def disconnect(self):
        """
        Disconnect from the MQTT broker.

        Publishes offline availability before disconnecting.
        """
        if not self._client:
            return

        try:
            # Publish offline status
            self.publish_availability(online=False)

            # Give time for message to be sent
            time.sleep(0.1)

            # Stop network loop and disconnect
            self._client.loop_stop()
            self._client.disconnect()

            logger.info("Disconnected from MQTT broker")

        except Exception as e:
            logger.error(f"Error during MQTT disconnect: {e}")

        finally:
            with self._lock:
                self._connected = False
            self._client = None
