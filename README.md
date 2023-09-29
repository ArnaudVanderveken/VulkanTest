# VulkanTest
This project was my first time working with Vulkan. I implemented the tutorial available on vulkan-tutorial.com in a new project, separate from any existing project, in order to focus on understanding how Vulkan works and link elements to my current knowledge of graphics programming and other APIs like DirectX 11 and 12. I skipped the last chapter about compute shaders, as it was going out of scope for this project.

All credits to the website vulkan-tutorial.com for their code and the resources they used.

More deatails about this project can be found on my personal website: www.arnaudvanderveken.com/projects/vulkan-renderer

## Installation
Make sure to use Visual Studio 2022.
After cloning the repository, you will need to install the VulkanSDK on your computer. Make sure to use version 1.3.250.1 or later.
For ease of installation, use version 1.3.250.1 and install it at the root of your C drive. If you use another version of the VulkanSDK or you installed it in another location, you will have to update the paths inside the visual studio project's properties ("C/C++ -> General -> Additional Include Directories" and "Linker -> General -> Additional Library Directories").
