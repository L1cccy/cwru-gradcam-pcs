-- pandoc-figures.lua: wrap images in figure environment with captions
function Figure(img)
  -- already a figure, do nothing
  return nil
end

function Image(el)
  -- Get the alt text as caption
  local caption = el.caption and el.caption[1] and el.caption[1].text or ""
  if caption == "" then
    return nil
  end
  
  -- Wrap inline image in a figure div
  local fig = pandoc.Div({
    pandoc.RawBlock("latex", "\\begin{figure}[htbp]\n\\centering"),
    pandoc.Para({el}),
    pandoc.RawBlock("latex", "\\caption{" .. caption .. "}\n\\end{figure}")
  }, {})
  
  return fig
end
