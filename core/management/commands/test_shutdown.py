"""
Management command to test graceful shutdown behavior.
"""
from django.core.management.base import BaseCommand
import signal
import time
import os


class Command(BaseCommand):
    help = 'Test graceful shutdown by sending SIGTERM to current process'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--delay',
            type=int,
            default=5,
            help='Seconds to wait before sending SIGTERM (default: 5)',
        )
    
    def handle(self, *args, **options):
        delay = options['delay']
        pid = os.getpid()
        
        self.stdout.write(self.style.SUCCESS(f'Process PID: {pid}'))
        self.stdout.write(self.style.WARNING(f'Will send SIGTERM in {delay} seconds...'))
        
        # Simulate some work
        for i in range(delay, 0, -1):
            self.stdout.write(f'  {i}...')
            time.sleep(1)
        
        self.stdout.write(self.style.WARNING('Sending SIGTERM now!'))
        os.kill(pid, signal.SIGTERM)
        
        # This should not be reached if graceful shutdown works
        time.sleep(2)
        self.stdout.write(self.style.ERROR('Still running after SIGTERM - graceful shutdown may not be working'))
