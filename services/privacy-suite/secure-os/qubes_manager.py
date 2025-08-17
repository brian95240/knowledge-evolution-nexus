#!/usr/bin/env python3
"""
K.E.N. Privacy Suite - Qubes OS Integration Manager
Automated disposable VM creation and management for secure operations
"""

import asyncio
import subprocess
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import tempfile
import os

@dataclass
class QubesVM:
    """Qubes VM configuration and state"""
    name: str
    template: str
    label: str
    vm_type: str  # 'disposable', 'app', 'template'
    netvm: Optional[str] = None
    memory: int = 1024
    vcpus: int = 1
    created_at: Optional[str] = None
    status: str = 'stopped'
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class QubesVMManager:
    """
    Qubes OS VM orchestration for K.E.N. privacy operations
    Manages disposable VMs with network isolation and secure cleanup
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        self.active_vms: Dict[str, QubesVM] = {}
        
        # Default configurations
        self.default_template = self.config.get('default_template', 'fedora-39-dvm')
        self.privacy_template = self.config.get('privacy_template', 'whonix-ws-17-dvm')
        self.tor_gateway = self.config.get('tor_gateway', 'sys-whonix')
        
        # VM naming convention
        self.vm_prefix = "ken-privacy"
        self.vm_counter = 0
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Qubes operations"""
        logger = logging.getLogger('QubesVMManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def check_qubes_availability(self) -> bool:
        """Check if Qubes OS is available and accessible"""
        try:
            result = await asyncio.create_subprocess_exec(
                'qvm-ls', '--raw-data',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                self.logger.info("Qubes OS detected and accessible")
                return True
            else:
                self.logger.warning("Qubes OS not available or not accessible")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking Qubes availability: {str(e)}")
            return False
    
    def _generate_vm_name(self, purpose: str = "general") -> str:
        """Generate unique VM name"""
        self.vm_counter += 1
        timestamp = int(time.time())
        return f"{self.vm_prefix}-{purpose}-{timestamp}-{self.vm_counter}"
    
    async def create_disposable_vm(self, 
                                 purpose: str = "privacy",
                                 use_tor: bool = True,
                                 memory_mb: int = 2048,
                                 vcpus: int = 2) -> Optional[QubesVM]:
        """
        Create disposable VM for privacy operations
        """
        try:
            vm_name = self._generate_vm_name(purpose)
            template = self.privacy_template if use_tor else self.default_template
            netvm = self.tor_gateway if use_tor else 'sys-firewall'
            
            # Create disposable VM
            create_cmd = [
                'qvm-create',
                '--class', 'DispVM',
                '--template', template,
                '--label', 'red',
                vm_name
            ]
            
            result = await asyncio.create_subprocess_exec(
                *create_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                self.logger.error(f"Failed to create VM: {stderr.decode()}")
                return None
            
            # Configure VM
            await self._configure_vm(vm_name, netvm, memory_mb, vcpus)
            
            # Create VM object
            vm = QubesVM(
                name=vm_name,
                template=template,
                label='red',
                vm_type='disposable',
                netvm=netvm,
                memory=memory_mb,
                vcpus=vcpus
            )
            
            self.active_vms[vm_name] = vm
            self.logger.info(f"Created disposable VM: {vm_name}")
            
            return vm
            
        except Exception as e:
            self.logger.error(f"Error creating disposable VM: {str(e)}")
            return None
    
    async def _configure_vm(self, vm_name: str, netvm: str, memory_mb: int, vcpus: int):
        """Configure VM settings"""
        try:
            # Set network VM
            await asyncio.create_subprocess_exec(
                'qvm-prefs', vm_name, 'netvm', netvm,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Set memory
            await asyncio.create_subprocess_exec(
                'qvm-prefs', vm_name, 'memory', str(memory_mb),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Set VCPUs
            await asyncio.create_subprocess_exec(
                'qvm-prefs', vm_name, 'vcpus', str(vcpus),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.logger.info(f"Configured VM {vm_name}: {memory_mb}MB RAM, {vcpus} vCPUs, netvm={netvm}")
            
        except Exception as e:
            self.logger.error(f"Error configuring VM {vm_name}: {str(e)}")
    
    async def start_vm(self, vm_name: str) -> bool:
        """Start VM and wait for it to be ready"""
        try:
            result = await asyncio.create_subprocess_exec(
                'qvm-start', vm_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                # Wait for VM to be fully started
                await self._wait_for_vm_ready(vm_name)
                
                if vm_name in self.active_vms:
                    self.active_vms[vm_name].status = 'running'
                
                self.logger.info(f"Started VM: {vm_name}")
                return True
            else:
                self.logger.error(f"Failed to start VM {vm_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting VM {vm_name}: {str(e)}")
            return False
    
    async def _wait_for_vm_ready(self, vm_name: str, timeout: int = 60):
        """Wait for VM to be fully ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = await asyncio.create_subprocess_exec(
                    'qvm-run', vm_name, 'echo "ready"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode == 0 and b"ready" in stdout:
                    self.logger.info(f"VM {vm_name} is ready")
                    return True
                    
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        self.logger.warning(f"VM {vm_name} may not be fully ready after {timeout}s")
        return False
    
    async def execute_in_vm(self, vm_name: str, command: str, user: str = 'user') -> Dict[str, Any]:
        """Execute command in VM and return result"""
        try:
            cmd = ['qvm-run', '--user', user, vm_name, command]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            return {
                'returncode': result.returncode,
                'stdout': stdout.decode(),
                'stderr': stderr.decode(),
                'success': result.returncode == 0
            }
            
        except Exception as e:
            self.logger.error(f"Error executing command in VM {vm_name}: {str(e)}")
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
    
    async def copy_to_vm(self, vm_name: str, local_path: str, vm_path: str) -> bool:
        """Copy file to VM"""
        try:
            result = await asyncio.create_subprocess_exec(
                'qvm-copy-to-vm', vm_name, local_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                self.logger.info(f"Copied {local_path} to VM {vm_name}")
                return True
            else:
                self.logger.error(f"Failed to copy file to VM: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error copying file to VM: {str(e)}")
            return False
    
    async def install_packages_in_vm(self, vm_name: str, packages: List[str]) -> bool:
        """Install packages in VM"""
        try:
            package_list = ' '.join(packages)
            command = f"sudo dnf install -y {package_list}"
            
            result = await self.execute_in_vm(vm_name, command)
            
            if result['success']:
                self.logger.info(f"Installed packages in VM {vm_name}: {packages}")
                return True
            else:
                self.logger.error(f"Failed to install packages: {result['stderr']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing packages in VM: {str(e)}")
            return False
    
    async def setup_privacy_environment(self, vm_name: str) -> bool:
        """Setup complete privacy environment in VM"""
        try:
            # Install required packages
            privacy_packages = [
                'tor', 'torbrowser-launcher', 'python3-pip',
                'firefox', 'chromium', 'curl', 'wget'
            ]
            
            if not await self.install_packages_in_vm(vm_name, privacy_packages):
                return False
            
            # Install Python packages for automation
            python_packages = "selenium playwright requests aiohttp"
            pip_command = f"pip3 install --user {python_packages}"
            
            result = await self.execute_in_vm(vm_name, pip_command)
            if not result['success']:
                self.logger.error(f"Failed to install Python packages: {result['stderr']}")
                return False
            
            # Configure Tor browser
            tor_config = """
# Tor configuration for K.E.N. privacy operations
SocksPort 9050
ControlPort 9051
CookieAuthentication 1
"""
            
            # Write Tor config
            config_command = f'echo "{tor_config}" | sudo tee /etc/tor/torrc.ken'
            await self.execute_in_vm(vm_name, config_command)
            
            self.logger.info(f"Setup privacy environment in VM: {vm_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up privacy environment: {str(e)}")
            return False
    
    async def destroy_vm(self, vm_name: str, force: bool = True) -> bool:
        """Destroy VM and cleanup"""
        try:
            # Stop VM first
            await asyncio.create_subprocess_exec(
                'qvm-shutdown', '--force' if force else '', vm_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait a moment for shutdown
            await asyncio.sleep(2)
            
            # Remove VM
            result = await asyncio.create_subprocess_exec(
                'qvm-remove', '--force', vm_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                if vm_name in self.active_vms:
                    del self.active_vms[vm_name]
                
                self.logger.info(f"Destroyed VM: {vm_name}")
                return True
            else:
                self.logger.error(f"Failed to destroy VM {vm_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error destroying VM {vm_name}: {str(e)}")
            return False
    
    async def cleanup_all_vms(self):
        """Cleanup all active VMs"""
        for vm_name in list(self.active_vms.keys()):
            await self.destroy_vm(vm_name)
        
        self.logger.info("Cleaned up all active VMs")
    
    async def get_vm_status(self, vm_name: str) -> Dict[str, Any]:
        """Get VM status and resource usage"""
        try:
            result = await asyncio.create_subprocess_exec(
                'qvm-ls', '--raw-data', vm_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                # Parse qvm-ls output
                lines = stdout.decode().strip().split('\n')
                if len(lines) >= 2:
                    headers = lines[0].split('|')
                    values = lines[1].split('|')
                    
                    status_data = dict(zip(headers, values))
                    return {
                        'name': vm_name,
                        'status': status_data.get('STATE', 'unknown'),
                        'memory': status_data.get('MEM', '0'),
                        'template': status_data.get('TEMPLATE', ''),
                        'netvm': status_data.get('NETVM', ''),
                        'available': True
                    }
            
            return {'name': vm_name, 'available': False}
            
        except Exception as e:
            self.logger.error(f"Error getting VM status: {str(e)}")
            return {'name': vm_name, 'available': False, 'error': str(e)}

# Integration functions for K.E.N.
async def ken_create_privacy_vm(purpose: str = "privacy", use_tor: bool = True) -> Optional[str]:
    """Create privacy VM for K.E.N. operations"""
    manager = QubesVMManager()
    
    if not await manager.check_qubes_availability():
        return None
    
    vm = await manager.create_disposable_vm(purpose=purpose, use_tor=use_tor)
    if not vm:
        return None
    
    if await manager.start_vm(vm.name):
        await manager.setup_privacy_environment(vm.name)
        return vm.name
    
    return None

async def ken_execute_in_privacy_vm(vm_name: str, command: str) -> Dict[str, Any]:
    """Execute command in privacy VM"""
    manager = QubesVMManager()
    return await manager.execute_in_vm(vm_name, command)

async def ken_cleanup_privacy_vm(vm_name: str) -> bool:
    """Cleanup privacy VM"""
    manager = QubesVMManager()
    return await manager.destroy_vm(vm_name)

if __name__ == "__main__":
    # Example usage
    async def main():
        manager = QubesVMManager()
        
        # Check availability
        if await manager.check_qubes_availability():
            print("Qubes OS is available")
            
            # Create privacy VM
            vm = await manager.create_disposable_vm("test", use_tor=True)
            if vm:
                print(f"Created VM: {vm.name}")
                
                # Start and setup
                if await manager.start_vm(vm.name):
                    await manager.setup_privacy_environment(vm.name)
                    
                    # Test command
                    result = await manager.execute_in_vm(vm.name, "whoami")
                    print(f"Command result: {result}")
                    
                    # Cleanup
                    await manager.destroy_vm(vm.name)
        else:
            print("Qubes OS not available - running in simulation mode")
    
    asyncio.run(main())

