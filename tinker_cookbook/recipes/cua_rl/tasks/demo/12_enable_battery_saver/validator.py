from __future__ import annotations

import logging

from ....adb import AdbClient

logger = logging.getLogger(__name__)


class Task12Validator:
    def validate(self, adb_client: AdbClient) -> bool:
        """
        Validate that battery saver mode is enabled.
        
        This validator tries multiple shell-based methods and logs detailed diagnostics.
        If all methods fail, treat it as a signal that the environment or command execution
        is misbehaving (capture logs for debugging) rather than silently relaxing validation.
        """
        logger.info("=" * 80)
        logger.info("[Task12Validator] Starting battery saver validation")
        logger.info("=" * 80)
        
        # Method 1: Check settings get global low_power (simplest method)
        try:
            logger.info("[Task12Validator] Method 1: Checking 'settings get global low_power'")
            cmd1 = "settings get global low_power"
            output1 = adb_client._run("shell", cmd1, capture_output=True)
            value1 = output1.strip()
            
            logger.info(f"[Task12Validator] Command: {cmd1}")
            logger.info(f"[Task12Validator] Raw output: {repr(output1)}")
            logger.info(f"[Task12Validator] Stripped value: {repr(value1)}")
            logger.info(f"[Task12Validator] Value type: {type(value1)}, length: {len(value1)}")
            
            # Battery saver is enabled when low_power = 1
            if value1 == "1":
                logger.info("[Task12Validator] ✓ Method 1 SUCCESS: Battery saver enabled (low_power=1)")
                return True
            elif value1 == "0":
                logger.warning("[Task12Validator] ✗ Method 1: Battery saver disabled (low_power=0)")
            else:
                logger.warning(f"[Task12Validator] ✗ Method 1: Unexpected value {repr(value1)}")
        except Exception as e:
            logger.error(f"[Task12Validator] ✗ Method 1 EXCEPTION: {e}")
        
        # Method 2: Check settings list global (no grep; piping is not reliable)
        try:
            logger.info("[Task12Validator] Method 2: Checking 'settings list global' for low_power keys")
            cmd2 = "settings list global"
            output2 = adb_client._run("shell", cmd2, capture_output=True)
            
            logger.info(f"[Task12Validator] Output length: {len(output2)} chars")
            logger.info(f"[Task12Validator] Output preview: {output2[:1000]}")
            
            # If we find low_power=1, battery saver is enabled
            if "low_power=1" in output2 or "low_power_mode=1" in output2:
                logger.info("[Task12Validator] ✓ Method 2 SUCCESS: Found low_power=1")
                return True
            elif "low_power=0" in output2 or "low_power_mode=0" in output2:
                logger.warning("[Task12Validator] ✗ Method 2: Found low_power=0")
        except Exception as e:
            logger.error(f"[Task12Validator] ✗ Method 2 EXCEPTION: {e}")
        
        # Method 3: Check dumpsys battery
        try:
            logger.info("[Task12Validator] Method 3: Checking 'dumpsys battery'")
            cmd3 = "dumpsys battery"
            output3 = adb_client._run("shell", cmd3, capture_output=True)
            
            logger.info(f"[Task12Validator] Output length: {len(output3)} chars")
            if output3:
                logger.info(f"[Task12Validator] Output preview: {output3[:500]}")
                
                # Common indicators
                indicators = [
                    ("powered by battery saver", True),
                    ("battery saver: on", True),
                    ("power save mode: on", True),
                    ("low_power: 1", True),
                    ("battery saver: off", False),
                    ("power save mode: off", False),
                    ("low_power: 0", False),
                ]
                
                for indicator, expected_state in indicators:
                    if indicator in output3.lower():
                        if expected_state:
                            logger.info(f"[Task12Validator] ✓ Method 3 SUCCESS: Found '{indicator}'")
                            return True
                        else:
                            logger.warning(f"[Task12Validator] ✗ Method 3: Found '{indicator}'")
                            break
        except Exception as e:
            logger.error(f"[Task12Validator] ✗ Method 3 EXCEPTION: {e}")
        
        # Method 4: Check dumpsys power
        try:
            logger.info("[Task12Validator] Method 4: Checking 'dumpsys power'")
            cmd4 = "dumpsys power"
            output4 = adb_client._run("shell", cmd4, capture_output=True)
            
            logger.info(f"[Task12Validator] Output length: {len(output4)} chars")
            
            if output4:
                # Look for power save related lines
                relevant_lines = [
                    line for line in output4.split('\n') 
                    if any(kw in line.lower() for kw in ['power save', 'battery save', 'low power'])
                ]
                
                logger.info(f"[Task12Validator] Found {len(relevant_lines)} relevant lines")
                for line in relevant_lines[:10]:
                    logger.info(f"[Task12Validator]   {line.strip()}")
                
                # Check for enabled state
                if any('power save=true' in line.lower() or 'low_power_mode=true' in line.lower() 
                       for line in relevant_lines):
                    logger.info("[Task12Validator] ✓ Method 4 SUCCESS: Power save enabled")
                    return True
        except Exception as e:
            logger.error(f"[Task12Validator] ✗ Method 4 EXCEPTION: {e}")
        
        # All methods failed
        logger.info("=" * 80)
        logger.warning("[Task12Validator] ⚠ VALIDATION INCONCLUSIVE")
        logger.warning("[Task12Validator] All shell validation methods failed or returned empty.")
        logger.warning("[Task12Validator] The UI may have been successfully toggled, but we could not verify via shell.")
        logger.warning("[Task12Validator] Please use these logs to debug command execution / permissions in the environment.")
        logger.info("=" * 80)
        return False
