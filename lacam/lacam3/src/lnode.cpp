#include "../include/lnode.hpp"

int LNode::COUNT = 0;

LNode::LNode() : who(), where(), depth(0) { ++COUNT; }

LNode::LNode(LNode *parent, int i, Vertex *v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
{
  ++COUNT;
  who.push_back(i);
  where.push_back(v);
}

LNode::~LNode(){};
