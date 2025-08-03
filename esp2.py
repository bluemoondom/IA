# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:38:56 2023

@author: dominika
"""

import asyncio
from kasa import SmartPlug

async def main():
    p = SmartPlug("192.168.1.16")
    await p.update()
    print(p.alias)
    await p.turn_on()
    #await p.turn_off()
    

asyncio.run(main())