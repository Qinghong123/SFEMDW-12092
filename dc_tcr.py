"""dc_tcr.py defines the DC DETCR utilities for XsRisc setup.
This only applies to 48 chn platform.
ToDO: this is to be replaced by ESF API SFEMDW-9826

Copyright (C) Seagate Technology LLC, 2021. All rights reserved."
"""

# Standar Python Modules
import time
import math

# User script modules
from component.pattern import Pattern
from legacy.drivers.preamp import preamp
from system.wrap import chn

SATURATION = 9999  # ADC reading is 255
STARVATION = -9999  # ADC reading is 0


class DctcrUtilsError(Exception): pass


class DctcrUtils:
    """Defines the XsRisc utilities for DC DETCR PTC."""

    # Scratch pad memory definition
    DIGIT_WRITE = 0x011  # Serializer write Preamp DigOn 1 command
    DIGIT_READ = 0x012
    DIGIT_MASK = 0x013
    ADC_READ = 0x014  # Serializer Read Preamp DSTR command (ADC)
    HEAT_WRITE = 0x015  # Serializer write Preamp ReHeat command with 0 heat

    DRDP_HEAT = 0x016  # ReHeat DACs in lower 8 bits
    DRDP_START = 0x017  # Switch to turn on the sampling process,
    # turned into 0 once the process is done
    OSC_SAMPLES = 0x018  # of oscillation samples per 1/2 cycle <=30; default 16
    SAMPLE_COUNT = 0x019  # Heat oscillation wave start address
    WARMUP_LENGTH = 0x01A  # Address for number of setors to wait before sampling start
    WARMUP_OFFSET = 0x01B  # Address for number of DACs to offset from DRDP_HEAT during warm up
    OFFSET_DIRECTION = 0x01C  # Address for WARMUP_OFFSET direciton: 0=down 1=up
    OSC_WAVE = 0x01D  # Heat oscillation wave start address

    DBG1 = 0x035
    ADC_DATA = 0x038  # ADC data section: = 0x02A to = 0x0FF => 214 32-bit registers
    MAX_WAVE = 56   # Maximum number of sine wave length
    BLOCK_SIZE = 800    # Memory block size for samples

    def __init__(self):
        # Initialize settings
        self.acquire_timeout = 10

    @staticmethod
    def getOscWave(wave_samples, peak):
        """Returns a list of Heater DACs that matches the half of a sine wave
        oscillation pattern: 0 -> peak -> 0
        Parameters
        ----------
        wave_samples: int
            the length of the Sine wave
        peak: int
            the heat Dacs of Sine wave peak
        """
        result = []
        pi = math.pi
        sin = math.sin
        n = wave_samples
        for i in range(n // 2):
            # 0 -> peak -> 0
            x = sin(2 * pi * i / n) * peak
            result.append(int(x + 0.5))
        print(f'Osc Wave = {result}')
        return result

    def setupAsm(self, heads, asm_file, heater, nrev, wu_rev, wu_offset,
                 wave_samples, peak):
        """ Sets the dRdP assembler scratch pad memory fields"""
        self.loadAsm(asm_file)
        # Check if dRdP assembler is running
        risc_no = preamp.get_preamp(0)
        v = int(chn.ReadXSRiscMemory('%d,0x%x' % (risc_no, self.DBG1)))
        if v == 8:
            print('dRdP asm is running!')
        else:
            raise DctcrUtilsError('XsRISC error: dRdP asm not started!')
        # Wait a half second for SPI to be ready
        time.sleep(0.5)
        # Register numbers and write mask for DigOn field
        reg_dig_no, mask = preamp.get_reg('DigOn')
        reg_dstr_no = preamp.get_reg('DSTR')[0]
        if heater == 'writer':
            reg_heater_no = preamp.get_reg('ReHeat')[0]  # writer mode
        else:
            reg_heater_no = preamp.get_reg('RDHT_R')[0]  # reader mode
        # Calculate total samples
        nsector = Pattern().getParams('number_of_sectors')
        size = int(nsector * nrev + 0.5)
        sample_size = min(size, self.BLOCK_SIZE)
        # Calculate warm up sectors
        wu_length = int(wu_rev * nsector + 0.5)
        if wu_offset > 0:
            wu_direction = 1
        else:
            wu_direction = 0
        wu_offset = abs(wu_offset)
        sine_wave = self.getOscWave(wave_samples, peak)
        # For each head, set its dRdP scratch pad memory settings
        for head in heads:
            risc_no = preamp.get_preamp(head)
            # Disable asm dRdP
            chn.WriteXSRiscMemory('%u,%u,0' % (risc_no, self.DRDP_START))
            value = int(chn.ReadPreampReg('%u,%u,0' % (risc_no, reg_dig_no)))
            # Reg value for setting DigOn to 1
            value = value | mask
            # Fixed address for LSI preamp
            spi_command = int(chn.EncodeSpiWrite('%u,%u' % (reg_dig_no, value)))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.DIGIT_WRITE, spi_command))

            spi_command = int(chn.EncodeSpiRead('%u' % reg_dig_no))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.DIGIT_READ, spi_command))

            spi_command = int(chn.EncodeSpiRead('%u' % reg_dstr_no))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.ADC_READ, spi_command))

            spi_command = int(chn.EncodeSpiWrite('%u,0' % reg_heater_no))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.HEAT_WRITE, spi_command))

            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.DIGIT_MASK, mask))
            # Set heat oscillation parameters
            chn.WriteXSRiscMemory('%u,%u,0' % (risc_no, self.DRDP_HEAT))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.SAMPLE_COUNT, sample_size))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.OSC_SAMPLES, wave_samples / 2))
            chn.WriteXSRiscMemory('%u,%u,%u' %
                                  (risc_no, self.WARMUP_LENGTH, wu_length))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.WARMUP_OFFSET, wu_offset))
            chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.OFFSET_DIRECTION, wu_direction))
            for i, value in enumerate(sine_wave):
                chn.WriteXSRiscMemory('%u,%u,%u' % (risc_no, self.OSC_WAVE + i, value))
        print(f'DC DETCR XsRisc setup finished: heater={heater}')

    def setHeater(self, heads, heater):
        """Sets the heater mode to be used in XsRisc.
        Parameters
        ----------
        heads: list
            the list of heads to use
        heater: str
            name of the heater, can be 'writer' or 'reader'

        """
        if heater == 'writer':
            reg_heater_no = preamp.get_reg('ReHeat')[0]
        elif heater == 'reader':
            reg_heater_no = preamp.gete_reg('RDHT_R')[0]
        else:
            raise DctcrUtilsError(f'Invalid heater name: {heater}')
        for head in heads:
            risc_no = preamp.get_preamp(head)
            spi_command = int(chn.WritePreampRegisterGenerateValueToWrite(
                '%u,%u,0' % (risc_no, reg_heater_no)))
            chn.WriteXSRiscMemory('%u,%u,%u' %
                                  (risc_no, self.HEAT_WRITE, spi_command))
        print(f'DC DETCR heater set to {heater} heater!')

    def loadAsm(self, asm_file):
        """Load the given assembler file to all heads. """
        print('Loading assembler file %s' % asm_file)
        chn.AperioUpdateProgram('%s,All' % asm_file)
