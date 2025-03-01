1. _pages: Contain files such as 
  -- podcasts.html 
  -- teaching.html 
  -- software.html
  -- publication.html 
  -- about.md has the blurb about me. 
  -- All this is in _pages.

2. _data: Contains content
  -- software.yml
  -- podcsts.yml
  -- teaching.yml
  -- talks.yml
  -- navigation.yml
  
3. How to do a blogpost
  -- Create an RMD file in the _Rmd folder
  -- Copy yml header from previous RMD
  -- output: 
  html_document:
    fig_path: "/assets/images/posts/"
    self_contained: true
    keep_md: true
  -- In R set up
  knitr::opts_chunk$set(
  fig.path = "../assets/images/posts/FOLDER_TO_SAVE_FILES/",  # Where to save files
  fig.cap = " ",
  fig.align = 'center'
  )
  -- After knitting, open the .md file and replace "../assets" with "/assets" in figures (VERY IMPORTANT)
  -- After knitting, move .md file to _posts folder *********** (VERY IMPORTANT)
  
4. How to add a talk
  -- Go to _data folder (at the root)
  -- Open talks.yml
  -- Add the talk on top

5. How to add a publication
  -- Go to _publications folder (at the root)
  -- Add a .md file for the publication
  -- yaml on top followed by APA style citation
  -- Update paper url 
